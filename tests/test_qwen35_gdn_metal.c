/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE)
 * gated-delta-net (linear attention) Metal decode kernel.
 *
 * This is the GPU sibling of tests/test_qwen35_gdn.c.  It feeds a synthetic
 * f32 fixture through `ds4_gpu_qwen35_gdn_decode` and against an independent
 * scalar reference of the same math: the fused attn_qkv conv + silu, the q/k
 * split and per-head L2 norm, the 2:1 value-head broadcast (n_k=16, n_v=32),
 * the folded `g = exp(a_log * softplus(alpha + dt_bias))` decay, the delta-rule
 * recurrence, the RMSNormGated * silu(z) output gate and the ssm_out output
 * projection.
 *
 * The projection matmuls (attn_qkv, attn_gate, alpha, beta) are not the
 * kernel's job -- the graph feeds them as dense matmuls -- so the test projects
 * on the CPU first and hands the kernel exactly what the graph would.
 *
 * The naive reference is a plain scalar implementation, so a silently-dropped
 * conv, decay, head broadcast or silu(z) gate cannot pass; Task 1's four
 * plants were run against it (see the report).
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-gdn-metal
 *   ./external/ds4/tests/test_qwen35_gdn_metal
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4_gpu.h"

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

/* Ornith-1.5-35B-A3B (DS4_SHAPE_ORNITH15) gated-delta-net geometry. */
enum {
    N_EMBD      = 2048,
    N_K         = 16,     /* key heads        */
    D_K         = 128,    /* key head dim     */
    N_V         = 32,     /* value heads      */
    D_V         = 128,    /* value head dim   */
    D_INNER     = 4096,   /* N_V * D_V        */
    CONV_K      = 4,      /* conv kernel      */
    CONV_DIM    = 2 * N_K * D_K + N_V * D_V, /* 8192 */
    N_ROWS      = 3,
    REC_SIZE    = N_V * D_V * D_K,           /* 524288 */
    CONV_STATE  = (CONV_K - 1) * CONV_DIM,   /* 24576  */

    /* Weight slots in the synthetic model map, 4096-byte aligned. */
    CONV1D_OFFSET  = 0,                        /* CONV_DIM * CONV_K floats */
    A_LOG_OFFSET   = CONV_DIM * CONV_K * 4u,
    DT_BIAS_OFFSET = A_LOG_OFFSET + 4096u,
    NORM_OFFSET    = DT_BIAS_OFFSET + 4096u,
    SSM_OUT_OFFSET = NORM_OFFSET + 4096u,
    MODEL_BYTES    = 64u * 1024u * 1024u,
};

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

static void require_close(const char *what, float actual, float expected,
                          float tolerance) {
    if (!isfinite(actual) || fabsf(actual - expected) > tolerance) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tolerance %.9g)\n",
                what, actual, expected, tolerance);
        exit(1);
    }
}

/* Deterministic, zero-mean-ish synthetic values. */
static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

static float sigmoid1(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

static float silu1(float x) {
    return x * sigmoid1(x);
}

static float softplus1(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void matvec1(float *out, const float *weight, uint32_t in_dim,
                    uint32_t out_dim, const float *x) {
    for (uint32_t o = 0; o < out_dim; o++) {
        const float *row = weight + (uint64_t)o * in_dim;
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)row[i] * x[i];
        out[o] = (float)acc;
    }
}

/* Independent scalar reference for the kernel's whole pipeline.  `qkv`, `z`,
 * `alpha` and `beta` are the already-projected inputs the graph would hand the
 * kernel; `conv_state` and `rec` are the persistent per-layer state buffers. */
static void reference_core(const float *qkv, const float *z,
                           const float *alpha, const float *beta,
                           const float *conv1d, const float *a_log,
                           const float *dt_bias, const float *norm,
                           const float *ssm_out, uint32_t n_rows,
                           float *conv_state, float *rec, float *out) {
    float convout[CONV_DIM];
    float qn[N_V * D_K], kn[N_V * D_K];
    float o[N_V * D_V];
    float on[D_INNER];

    for (uint32_t t = 0; t < n_rows; t++) {
        const float *qkv_t = qkv + (uint64_t)t * CONV_DIM;
        for (uint32_t ch = 0; ch < CONV_DIM; ch++) {
            double acc = 0.0;
            for (uint32_t kk = 0; kk < CONV_K; kk++) {
                const float in = (kk < CONV_K - 1)
                                     ? conv_state[(uint64_t)kk * CONV_DIM + ch]
                                     : qkv_t[ch];
                acc += (double)in * conv1d[(uint64_t)ch * CONV_K + kk];
            }
            convout[ch] = silu1((float)acc);
        }
        for (uint32_t kk = 0; kk + 1 < CONV_K - 1; kk++)
            memcpy(conv_state + (uint64_t)kk * CONV_DIM,
                   conv_state + (uint64_t)(kk + 1) * CONV_DIM,
                   CONV_DIM * sizeof(float));
        memcpy(conv_state + (uint64_t)(CONV_K - 2) * CONV_DIM, qkv_t,
               CONV_DIM * sizeof(float));

        /* Per-head L2 norm, then repeat q/k to the value heads. */
        for (uint32_t h = 0; h < N_K; h++) {
            const float *qraw = convout + (uint64_t)h * D_K;
            double ss = 0.0;
            for (uint32_t d = 0; d < D_K; d++) ss += (double)qraw[d] * qraw[d];
            const float qinv = 1.0f / sqrtf((float)ss + 1e-6f);
            for (uint32_t vh = h; vh < N_V; vh += N_K)
                for (uint32_t d = 0; d < D_K; d++)
                    qn[(uint64_t)vh * D_K + d] =
                        qraw[d] * qinv * (1.0f / sqrtf((float)D_K));

            const float *kraw = convout + (uint64_t)N_K * D_K + (uint64_t)h * D_K;
            ss = 0.0;
            for (uint32_t d = 0; d < D_K; d++) ss += (double)kraw[d] * kraw[d];
            const float kinv = 1.0f / sqrtf((float)ss + 1e-6f);
            for (uint32_t vh = h; vh < N_V; vh += N_K)
                for (uint32_t d = 0; d < D_K; d++)
                    kn[(uint64_t)vh * D_K + d] = kraw[d] * kinv;
        }

        const float *vraw = convout + 2 * N_K * D_K;

        for (uint32_t vh = 0; vh < N_V; vh++) {
            const float gate =
                a_log[vh] * softplus1(alpha[(uint64_t)t * N_V + vh] +
                                      dt_bias[vh]);
            const float g = expf(gate);
            const float b = sigmoid1(beta[(uint64_t)t * N_V + vh]);

            float *s = rec + (uint64_t)vh * D_V * D_K;
            for (uint32_t i = 0; i < D_V * D_K; i++) s[i] *= g;

            float sk[D_V], dd[D_V];
            for (uint32_t dv = 0; dv < D_V; dv++) {
                double acc = 0.0;
                for (uint32_t dk = 0; dk < D_K; dk++)
                    acc += (double)s[(uint64_t)dv * D_K + dk] *
                           kn[(uint64_t)vh * D_K + dk];
                sk[dv] = (float)acc;
            }
            for (uint32_t dv = 0; dv < D_V; dv++)
                dd[dv] = (vraw[(uint64_t)vh * D_V + dv] - sk[dv]) * b;
            for (uint32_t dv = 0; dv < D_V; dv++)
                for (uint32_t dk = 0; dk < D_K; dk++)
                    s[(uint64_t)dv * D_K + dk] +=
                        kn[(uint64_t)vh * D_K + dk] * dd[dv];
            for (uint32_t dv = 0; dv < D_V; dv++) {
                double acc = 0.0;
                for (uint32_t dk = 0; dk < D_K; dk++)
                    acc += (double)s[(uint64_t)dv * D_K + dk] *
                           qn[(uint64_t)vh * D_K + dk];
                o[(uint64_t)vh * D_V + dv] = (float)acc;
            }
        }

        for (uint32_t vh = 0; vh < N_V; vh++) {
            double ss = 0.0;
            for (uint32_t dv = 0; dv < D_V; dv++)
                ss += (double)o[(uint64_t)vh * D_V + dv] *
                      o[(uint64_t)vh * D_V + dv];
            const float inv = 1.0f / sqrtf((float)(ss / D_V) + 1e-6f);
            for (uint32_t dv = 0; dv < D_V; dv++) {
                const uint64_t idx = (uint64_t)vh * D_V + dv;
                on[idx] = o[idx] * inv * norm[dv] *
                          silu1(z[(uint64_t)t * D_INNER + idx]);
            }
        }

        matvec1(out + (uint64_t)t * N_EMBD, ssm_out, D_INNER, N_EMBD, on);
    }
}

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    float *conv1d = (float *)(model + CONV1D_OFFSET);
    float *a_log = (float *)(model + A_LOG_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    float *norm = (float *)(model + NORM_OFFSET);
    float *ssm_out = (float *)(model + SSM_OUT_OFFSET);

    for (uint32_t ch = 0; ch < CONV_DIM; ch++)
        for (uint32_t kk = 0; kk < CONV_K; kk++)
            conv1d[(uint64_t)ch * CONV_K + kk] = synth(ch + 3u, kk, 0.35f);
    for (uint32_t h = 0; h < N_V; h++) {
        /* ssm_a is stored already folded as -exp(A_log): every value is
         * negative, and its magnitude sets the decay rate. */
        a_log[h] = -(0.4f + 0.1f * (float)(h % 5u));
        dt_bias[h] = 0.2f * (float)(h % 3u) - 0.2f;
    }
    for (uint32_t d = 0; d < D_V; d++)
        norm[d] = 0.75f + 0.0004f * (float)(d % 251u);
    for (uint32_t o = 0; o < N_EMBD; o++)
        for (uint32_t j = 0; j < D_INNER; j++)
            ssm_out[(uint64_t)o * D_INNER + j] = synth(o + 101u, j, 0.02f);

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    ds4_gpu_tensor *g_qkv = ds4_gpu_tensor_alloc((uint64_t)N_ROWS * CONV_DIM * sizeof(float));
    ds4_gpu_tensor *g_z = ds4_gpu_tensor_alloc((uint64_t)N_ROWS * D_INNER * sizeof(float));
    ds4_gpu_tensor *g_alpha = ds4_gpu_tensor_alloc((uint64_t)N_ROWS * N_V * sizeof(float));
    ds4_gpu_tensor *g_beta = ds4_gpu_tensor_alloc((uint64_t)N_ROWS * N_V * sizeof(float));
    ds4_gpu_tensor *g_out = ds4_gpu_tensor_alloc((uint64_t)N_ROWS * N_EMBD * sizeof(float));
    ds4_gpu_tensor *g_conv = ds4_gpu_tensor_alloc((uint64_t)CONV_STATE * sizeof(float));
    ds4_gpu_tensor *g_rec = ds4_gpu_tensor_alloc((uint64_t)REC_SIZE * sizeof(float));
    require_ok(g_qkv && g_z && g_alpha && g_beta && g_out && g_conv && g_rec,
               "tensor allocation");

    /* The projected inputs a real layer would produce from the hidden state.
     * Nearly identical rows keep the key/value vectors aligned across
     * positions, so the delta-rule state and its decay leave a large,
     * unmistakable signature from the second token on. */
    float *qkv = malloc((uint64_t)N_ROWS * CONV_DIM * sizeof(float));
    float *z = malloc((uint64_t)N_ROWS * D_INNER * sizeof(float));
    float *alpha = malloc((uint64_t)N_ROWS * N_V * sizeof(float));
    float *beta = malloc((uint64_t)N_ROWS * N_V * sizeof(float));
    require_ok(qkv && z && alpha && beta, "host input allocation");
    for (uint32_t t = 0; t < N_ROWS; t++) {
        for (uint32_t ch = 0; ch < CONV_DIM; ch++)
            qkv[(uint64_t)t * CONV_DIM + ch] =
                synth(ch + 1u, 17u, 0.05f) + 0.005f * (float)t;
        for (uint32_t j = 0; j < D_INNER; j++)
            z[(uint64_t)t * D_INNER + j] =
                synth(j + 5u, 29u, 0.5f) + 0.01f * (float)t;
        for (uint32_t h = 0; h < N_V; h++) {
            alpha[(uint64_t)t * N_V + h] = synth(h + 7u, 41u, 0.2f);
            beta[(uint64_t)t * N_V + h] = synth(h + 13u, 53u, 0.2f);
        }
    }
    require_ok(ds4_gpu_tensor_write(g_qkv, 0, qkv,
                                    (uint64_t)N_ROWS * CONV_DIM * sizeof(float)),
               "qkv write");
    require_ok(ds4_gpu_tensor_write(g_z, 0, z,
                                    (uint64_t)N_ROWS * D_INNER * sizeof(float)),
               "z write");
    require_ok(ds4_gpu_tensor_write(g_alpha, 0, alpha,
                                    (uint64_t)N_ROWS * N_V * sizeof(float)),
               "alpha write");
    require_ok(ds4_gpu_tensor_write(g_beta, 0, beta,
                                    (uint64_t)N_ROWS * N_V * sizeof(float)),
               "beta write");

    /* Single-token case from a zero state. */
    {
        static float st_ref[REC_SIZE], cv_ref[CONV_STATE];
        static float st_gpu[REC_SIZE], cv_gpu[CONV_STATE];
        memset(st_ref, 0, sizeof(st_ref));
        memset(cv_ref, 0, sizeof(cv_ref));
        memset(st_gpu, 0, sizeof(st_gpu));
        memset(cv_gpu, 0, sizeof(cv_gpu));
        float expect[N_EMBD], actual[N_EMBD];
        reference_core(qkv, z, alpha, beta, conv1d, a_log, dt_bias, norm,
                       ssm_out, 1, cv_ref, st_ref, expect);
        require_ok(ds4_gpu_tensor_fill_f32(g_conv, 0.0f, CONV_STATE), "conv clear");
        require_ok(ds4_gpu_tensor_fill_f32(g_rec, 0.0f, REC_SIZE), "rec clear");
        require_ok(ds4_gpu_qwen35_gdn_decode(
                       g_out, g_conv, g_rec, g_qkv, g_z, g_alpha, g_beta,
                       model, MODEL_BYTES, CONV1D_OFFSET, A_LOG_OFFSET,
                       DT_BIAS_OFFSET, NORM_OFFSET, SSM_OUT_OFFSET,
                       1, 1e-6f),
                   "gdn decode single token");
        require_ok(ds4_gpu_tensor_read(g_out, 0, actual, sizeof(actual)),
                   "output read");
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 gdn metal single token", actual[d], expect[d],
                          3e-3f);
    }

    /* Three-token call: the kernel carries the conv window and the recurrent
     * state across rows internally. */
    {
        static float st_ref[REC_SIZE], cv_ref[CONV_STATE];
        static float st_gpu[REC_SIZE], cv_gpu[CONV_STATE];
        memset(st_ref, 0, sizeof(st_ref));
        memset(cv_ref, 0, sizeof(cv_ref));
        memset(st_gpu, 0, sizeof(st_gpu));
        memset(cv_gpu, 0, sizeof(cv_gpu));
        float expect[(uint64_t)N_ROWS * N_EMBD];
        float actual[(uint64_t)N_ROWS * N_EMBD];
        reference_core(qkv, z, alpha, beta, conv1d, a_log, dt_bias, norm,
                       ssm_out, N_ROWS, cv_ref, st_ref, expect);
        require_ok(ds4_gpu_tensor_fill_f32(g_conv, 0.0f, CONV_STATE), "conv reset");
        require_ok(ds4_gpu_tensor_fill_f32(g_rec, 0.0f, REC_SIZE), "rec reset");
        require_ok(ds4_gpu_qwen35_gdn_decode(
                       g_out, g_conv, g_rec, g_qkv, g_z, g_alpha, g_beta,
                       model, MODEL_BYTES, CONV1D_OFFSET, A_LOG_OFFSET,
                       DT_BIAS_OFFSET, NORM_OFFSET, SSM_OUT_OFFSET,
                       N_ROWS, 1e-6f),
                   "gdn decode three tokens");
        require_ok(ds4_gpu_tensor_read(g_out, 0, actual, sizeof(actual)),
                   "output read");
        for (uint32_t t = 0; t < N_ROWS; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                require_close("qwen35 gdn metal three tokens",
                              actual[(uint64_t)t * N_EMBD + d],
                              expect[(uint64_t)t * N_EMBD + d], 3e-3f);
        require_ok(ds4_gpu_tensor_read(g_rec, 0, st_gpu,
                                       (uint64_t)REC_SIZE * sizeof(float)),
                   "recurrent state read");
        require_ok(ds4_gpu_tensor_read(g_conv, 0, cv_gpu,
                                       (uint64_t)CONV_STATE * sizeof(float)),
                   "conv state read");
        require_ok(max_abs_diff(st_ref, st_gpu, REC_SIZE) < 1e-4f,
                   "qwen35 gdn metal recurrent state matches");
        require_ok(max_abs_diff(cv_ref, cv_gpu, CONV_STATE) < 1e-6f,
                   "qwen35 gdn metal conv state matches");
    }

    /* State persistence across calls: one token, then one more token from the
     * carried state must equal the batched three-token reference. */
    {
        static float st_ref[REC_SIZE], cv_ref[CONV_STATE];
        memset(st_ref, 0, sizeof(st_ref));
        memset(cv_ref, 0, sizeof(cv_ref));
        float expect[(uint64_t)N_ROWS * N_EMBD];
        reference_core(qkv, z, alpha, beta, conv1d, a_log, dt_bias, norm,
                       ssm_out, N_ROWS, cv_ref, st_ref, expect);

        static float st_gpu[REC_SIZE], cv_gpu[CONV_STATE];
        memset(st_gpu, 0, sizeof(st_gpu));
        memset(cv_gpu, 0, sizeof(cv_gpu));
        float actual[(uint64_t)N_ROWS * N_EMBD];
        require_ok(ds4_gpu_tensor_fill_f32(g_conv, 0.0f, CONV_STATE), "conv clear");
        require_ok(ds4_gpu_tensor_fill_f32(g_rec, 0.0f, REC_SIZE), "rec clear");
        for (uint32_t t = 0; t < N_ROWS; t++) {
            /* Load row t; the conv/recurrent state persists across calls. */
            require_ok(ds4_gpu_tensor_write(g_qkv, 0, qkv + (uint64_t)t * CONV_DIM,
                                            (uint64_t)CONV_DIM * sizeof(float)),
                       "qkv row write");
            require_ok(ds4_gpu_tensor_write(g_z, 0, z + (uint64_t)t * D_INNER,
                                            (uint64_t)D_INNER * sizeof(float)),
                       "z row write");
            require_ok(ds4_gpu_tensor_write(g_alpha, 0, alpha + (uint64_t)t * N_V,
                                            (uint64_t)N_V * sizeof(float)),
                       "alpha row write");
            require_ok(ds4_gpu_tensor_write(g_beta, 0, beta + (uint64_t)t * N_V,
                                            (uint64_t)N_V * sizeof(float)),
                       "beta row write");
            require_ok(ds4_gpu_qwen35_gdn_decode(
                           g_out, g_conv, g_rec,
                           g_qkv, g_z, g_alpha, g_beta,
                           model, MODEL_BYTES, CONV1D_OFFSET, A_LOG_OFFSET,
                           DT_BIAS_OFFSET, NORM_OFFSET, SSM_OUT_OFFSET,
                           1, 1e-6f),
                       "gdn decode split call");
            require_ok(ds4_gpu_tensor_read(
                           g_out, 0, actual + (uint64_t)t * N_EMBD,
                           (uint64_t)N_EMBD * sizeof(float)),
                       "split output read");
        }
        for (uint32_t t = 0; t < N_ROWS; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                require_close("qwen35 gdn metal split call",
                              actual[(uint64_t)t * N_EMBD + d],
                              expect[(uint64_t)t * N_EMBD + d], 3e-3f);
    }

    free(beta);
    free(alpha);
    free(z);
    free(qkv);
    ds4_gpu_tensor_free(g_rec);
    ds4_gpu_tensor_free(g_conv);
    ds4_gpu_tensor_free(g_out);
    ds4_gpu_tensor_free(g_beta);
    ds4_gpu_tensor_free(g_alpha);
    ds4_gpu_tensor_free(g_z);
    ds4_gpu_tensor_free(g_qkv);
    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("qwen35 gated-delta-net Metal decode: PASS");
    return 0;
}
