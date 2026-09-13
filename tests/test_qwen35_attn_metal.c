/* Model-free tests for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE) Metal
 * full-attention support kernels: GQA FlashAttention, the weighted strided
 * per-head q/k RMSNorm, the Neox half-pair partial front RoPE, and the
 * sigmoid output gate.
 *
 * The CPU full-attention core (ds4.c, qwen35_attn_core) is the golden
 * reference:
 *   - the joint attn_q projection is [n_head][2*head_dim] with the query in
 *     the first head_dim of each head and the sigmoid gate in the second, so
 *     the query RMSNorm is weighted but strided by 2*head_dim;
 *   - K is contiguous per KV head, weighted by attn_k_norm;
 *   - partial RoPE is Neox half-pairing over the FIRST n_rot dims, pair
 *     (p, p + n_rot/2), theta = pos * base^(-2p/n_rot);
 *   - GQA maps query head h to KV head h / (n_head / n_head_kv);
 *   - the sigmoid gate multiplies the attention context elementwise.
 *
 * Each reference is a plain scalar implementation, so a silently-dropped
 * norm, RoPE, GQA grouping or gate cannot pass.  Task 2's plants were run
 * against these tests (see the report).
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-attn-metal
 *   ./external/ds4/tests/test_qwen35_attn_metal
 */
#include <float.h>
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

/* Ornith-1.5-35B-A3B (DS4_SHAPE_ORNITH15) full-attention geometry. */
enum {
    N_HEAD      = 16,
    N_HEAD_KV   = 2,
    HEAD_DIM    = 256,
    N_ROT       = 64,
    HALF        = N_ROT / 2,
    GROUP       = N_HEAD / N_HEAD_KV,
    Q_GATE_DIM  = 2 * HEAD_DIM,
    Q_STRIDE    = Q_GATE_DIM,       /* floats per query+gate head */
    KV_STRIDE   = HEAD_DIM,         /* floats per KV head */
    N_TOKENS    = 3,
    CACHE_CAP   = 8,
    Q_NORM_OFF  = 0,
    K_NORM_OFF  = 4096,
    MODEL_BYTES = 1u << 20,
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

static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

static float max_abs_diff(const float *a, const float *b, uint64_t n) {
    float m = 0.0f;
    for (uint64_t i = 0; i < n; i++) m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

/* Weighted RMSNorm of the first `n` floats of a strided head row. */
static void reference_head_norm(const float *in, float *out, const float *w,
                                uint32_t rows, uint32_t n, uint32_t stride) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *x = in + (uint64_t)r * stride;
        float *y = out + (uint64_t)r * stride;
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1e-6f);
        for (uint32_t i = 0; i < n; i++) y[i] = x[i] * scale * w[i];
    }
}

/* Neox half-pair partial RoPE over the first n_rot dims of each head. */
static void reference_rope(float *x, uint32_t n_head, uint32_t stride,
                           uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                           float freq_base) {
    const uint32_t half = n_rot / 2u;
    for (uint32_t t = 0; t < N_TOKENS; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *head = x + (uint64_t)t * n_head * stride + (uint64_t)h * stride;
            const float pos = (float)(pos0 + t);
            for (uint32_t p = 0; p < half; p++) {
                const float theta = pos * powf(freq_base, -2.0f * (float)p / (float)n_rot);
                const float c = cosf(theta);
                const float s = sinf(theta);
                const float x0 = head[p];
                const float x1 = head[p + half];
                head[p]        = x0 * c - x1 * s;
                head[p + half] = x0 * s + x1 * c;
            }
        }
    }
    (void)head_dim;
}

/* Naive GQA causal attention: query head h attends to KV head
 * h / GROUP with a softmax over keys <= its position. */
static void reference_gqa(const float *q, const float *k_cache,
                          const float *v_cache, float *out, uint32_t n_tokens,
                          uint32_t n_head, uint32_t n_head_kv,
                          uint32_t head_dim, uint32_t group) {
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            const uint32_t kv = h / group;
            const float *qh =
                q + ((uint64_t)t * n_head + h) * head_dim;
            float *oh =
                out + ((uint64_t)t * n_head + h) * head_dim;
            float scores[64];
            float max_score = -FLT_MAX;
            for (uint32_t s = 0; s <= t; s++) {
                const float *kh =
                    k_cache + ((uint64_t)s * n_head_kv + kv) * head_dim;
                double dot = 0.0;
                for (uint32_t d = 0; d < head_dim; d++)
                    dot += (double)qh[d] * kh[d];
                scores[s] = (float)dot * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }
            double denom = 0.0;
            for (uint32_t s = 0; s <= t; s++) {
                scores[s] = expf(scores[s] - max_score);
                denom += scores[s];
            }
            for (uint32_t d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (uint32_t s = 0; s <= t; s++) {
                    const float *vh =
                        v_cache + ((uint64_t)s * n_head_kv + kv) * head_dim;
                    acc += ((double)scores[s] / denom) * vh[d];
                }
                oh[d] = (float)acc;
            }
        }
    }
}

int main(void) {
    uint8_t *model = mmap(NULL, MODEL_BYTES, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANON, -1, 0);
    if (model == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    float *q_norm_w = (float *)(model + Q_NORM_OFF);
    float *k_norm_w = (float *)(model + K_NORM_OFF);
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        q_norm_w[d] = 0.8f + 0.0004f * (float)(d % 251u);
        k_norm_w[d] = 1.0f + 0.0003f * (float)(d % 199u);
    }

    require_ok(ds4_gpu_init(), "GPU initialization");
    require_ok(ds4_gpu_set_model_map(model, MODEL_BYTES), "model map registration");

    /* ---------------------------------------------------------------- (d) */
    /* Sigmoid output gate: elementwise against the scalar reference. */
    {
        enum { N = 4097 };
        float *in = malloc(N * sizeof(float));
        float *expect = malloc(N * sizeof(float));
        float *actual = malloc(N * sizeof(float));
        require_ok(in && expect && actual, "sigmoid host allocation");
        for (uint32_t i = 0; i < N; i++) {
            in[i] = -9.0f + 18.0f * (float)i / (float)(N - 1);
            expect[i] = 1.0f / (1.0f + expf(-in[i]));
        }
        ds4_gpu_tensor *gx = ds4_gpu_tensor_alloc((uint64_t)N * sizeof(float));
        ds4_gpu_tensor *go = ds4_gpu_tensor_alloc((uint64_t)N * sizeof(float));
        require_ok(gx && go, "sigmoid tensor allocation");
        require_ok(ds4_gpu_tensor_write(gx, 0, in, (uint64_t)N * sizeof(float)),
                   "sigmoid write");
        require_ok(ds4_gpu_sigmoid_tensor(go, gx, N), "sigmoid launch");
        require_ok(ds4_gpu_tensor_read(go, 0, actual,
                                       (uint64_t)N * sizeof(float)),
                   "sigmoid read");
        fprintf(stderr, "qwen35 attn metal: sigmoid max diff %.3g\n",
                max_abs_diff(actual, expect, N));
        for (uint32_t i = 0; i < N; i++)
            require_close("qwen35 sigmoid", actual[i], expect[i], 1e-5f);
        ds4_gpu_tensor_free(go);
        ds4_gpu_tensor_free(gx);
        free(actual);
        free(expect);
        free(in);
    }

    /* ---------------------------------------------------------------- (b) */
    /* Weighted per-head q/k RMSNorm over the interleaved q/gate layout. */
    {
        const uint32_t q_rows = N_TOKENS * N_HEAD;
        const uint32_t k_rows = N_TOKENS * N_HEAD_KV;
        const uint64_t q_count = (uint64_t)q_rows * Q_STRIDE;
        const uint64_t k_count = (uint64_t)k_rows * KV_STRIDE;
        float *q_in = malloc(q_count * sizeof(float));
        float *k_in = malloc(k_count * sizeof(float));
        float *q_exp = malloc(q_count * sizeof(float));
        float *k_exp = malloc(k_count * sizeof(float));
        float *q_act = malloc(q_count * sizeof(float));
        float *k_act = malloc(k_count * sizeof(float));
        require_ok(q_in && k_in && q_exp && k_exp && q_act && k_act,
                   "norm host allocation");
        for (uint64_t i = 0; i < q_count; i++)
            q_in[i] = synth((uint32_t)i, 11u, 0.6f);
        for (uint64_t i = 0; i < k_count; i++)
            k_in[i] = synth((uint32_t)i + 3u, 19u, 0.6f);
        memcpy(q_exp, q_in, q_count * sizeof(float));
        memcpy(k_exp, k_in, k_count * sizeof(float));
        /* Reference normalises only the first HEAD_DIM of each q row. */
        reference_head_norm(q_in, q_exp, q_norm_w, q_rows, HEAD_DIM, Q_STRIDE);
        reference_head_norm(k_in, k_exp, k_norm_w, k_rows, HEAD_DIM, KV_STRIDE);

        ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(q_count * sizeof(float));
        ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(k_count * sizeof(float));
        require_ok(gq && gk, "norm tensor allocation");
        require_ok(ds4_gpu_tensor_write(gq, 0, q_in, q_count * sizeof(float)),
                   "q write");
        require_ok(ds4_gpu_tensor_write(gk, 0, k_in, k_count * sizeof(float)),
                   "k write");
        require_ok(ds4_gpu_qwen35_head_rms_norm_tensor(
                       gq, model, MODEL_BYTES, Q_NORM_OFF, q_rows, HEAD_DIM,
                       (uint64_t)Q_STRIDE * sizeof(float), 1e-6f),
                   "q head norm");
        require_ok(ds4_gpu_qwen35_head_rms_norm_tensor(
                       gk, model, MODEL_BYTES, K_NORM_OFF, k_rows, HEAD_DIM,
                       (uint64_t)KV_STRIDE * sizeof(float), 1e-6f),
                   "k head norm");
        require_ok(ds4_gpu_tensor_read(gq, 0, q_act, q_count * sizeof(float)),
                   "q read");
        require_ok(ds4_gpu_tensor_read(gk, 0, k_act, k_count * sizeof(float)),
                   "k read");
        fprintf(stderr, "qwen35 attn metal: q/k norm max diff %.3g\n",
                fmaxf(max_abs_diff(q_act, q_exp, q_count),
                      max_abs_diff(k_act, k_exp, k_count)));
        for (uint64_t i = 0; i < q_count; i++)
            require_close("qwen35 q head norm", q_act[i], q_exp[i], 1e-5f);
        for (uint64_t i = 0; i < k_count; i++)
            require_close("qwen35 k head norm", k_act[i], k_exp[i], 1e-5f);

        /* The gate half of every q row must be untouched: the stride must not
         * let the normaliser wander into the interleaved gate. */
        for (uint32_t r = 0; r < q_rows; r++)
            for (uint32_t d = HEAD_DIM; d < Q_STRIDE; d++)
                require_close("qwen35 q norm leaves gate half",
                              q_act[(uint64_t)r * Q_STRIDE + d],
                              q_in[(uint64_t)r * Q_STRIDE + d], 0.0f);
        ds4_gpu_tensor_free(gk);
        ds4_gpu_tensor_free(gq);
        free(k_act); free(q_act); free(k_exp); free(q_exp);
        free(k_in); free(q_in);
    }

    /* ---------------------------------------------------------------- (c) */
    /* Neox half-pair partial front RoPE: dims [n_rot, head_dim) untouched. */
    {
        const uint32_t q_rows = N_TOKENS * N_HEAD;
        const uint32_t k_rows = N_TOKENS * N_HEAD_KV;
        const uint64_t q_count = (uint64_t)q_rows * Q_STRIDE;
        const uint64_t k_count = (uint64_t)k_rows * KV_STRIDE;
        float *q_in = malloc(q_count * sizeof(float));
        float *k_in = malloc(k_count * sizeof(float));
        float *q_exp = malloc(q_count * sizeof(float));
        float *k_exp = malloc(k_count * sizeof(float));
        float *q_act = malloc(q_count * sizeof(float));
        float *k_act = malloc(k_count * sizeof(float));
        require_ok(q_in && k_in && q_exp && k_exp && q_act && k_act,
                   "rope host allocation");
        for (uint64_t i = 0; i < q_count; i++)
            q_in[i] = synth((uint32_t)i, 23u, 0.6f);
        for (uint64_t i = 0; i < k_count; i++)
            k_in[i] = synth((uint32_t)i + 5u, 31u, 0.6f);
        memcpy(q_exp, q_in, q_count * sizeof(float));
        memcpy(k_exp, k_in, k_count * sizeof(float));
        reference_rope(q_exp, N_HEAD, Q_STRIDE, HEAD_DIM, N_ROT, 3, 1e7f);
        reference_rope(k_exp, N_HEAD_KV, KV_STRIDE, HEAD_DIM, N_ROT, 3, 1e7f);

        ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(q_count * sizeof(float));
        ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(k_count * sizeof(float));
        require_ok(gq && gk, "rope tensor allocation");
        require_ok(ds4_gpu_tensor_write(gq, 0, q_in, q_count * sizeof(float)),
                   "rope q write");
        require_ok(ds4_gpu_tensor_write(gk, 0, k_in, k_count * sizeof(float)),
                   "rope k write");
        require_ok(ds4_gpu_qwen35_rope_neox_front_tensor(
                       gq, N_TOKENS, N_HEAD, HEAD_DIM, N_ROT,
                       (uint64_t)Q_STRIDE * sizeof(float),
                       (uint64_t)N_HEAD * Q_STRIDE * sizeof(float),
                       3, 1e7f),
                   "q rope");
        require_ok(ds4_gpu_qwen35_rope_neox_front_tensor(
                       gk, N_TOKENS, N_HEAD_KV, HEAD_DIM, N_ROT,
                       (uint64_t)KV_STRIDE * sizeof(float),
                       (uint64_t)N_HEAD_KV * KV_STRIDE * sizeof(float),
                       3, 1e7f),
                   "k rope");
        require_ok(ds4_gpu_tensor_read(gq, 0, q_act, q_count * sizeof(float)),
                   "rope q read");
        require_ok(ds4_gpu_tensor_read(gk, 0, k_act, k_count * sizeof(float)),
                   "rope k read");
        fprintf(stderr, "qwen35 attn metal: rope max diff %.3g\n",
                fmaxf(max_abs_diff(q_act, q_exp, q_count),
                      max_abs_diff(k_act, k_exp, k_count)));
        for (uint64_t i = 0; i < q_count; i++)
            require_close("qwen35 q rope", q_act[i], q_exp[i], 1e-5f);
        for (uint64_t i = 0; i < k_count; i++)
            require_close("qwen35 k rope", k_act[i], k_exp[i], 1e-5f);

        /* Dims [n_rot, head_dim) of q and k, and the whole q gate half, are
         * never touched by the partial rotation. */
        for (uint32_t t = 0; t < N_TOKENS; t++) {
            for (uint32_t h = 0; h < N_HEAD; h++) {
                const float *in =
                    q_in + ((uint64_t)t * N_HEAD + h) * Q_STRIDE;
                const float *out =
                    q_act + ((uint64_t)t * N_HEAD + h) * Q_STRIDE;
                for (uint32_t d = N_ROT; d < HEAD_DIM; d++)
                    require_close("qwen35 q rope tail untouched", out[d], in[d], 0.0f);
                for (uint32_t d = HEAD_DIM; d < Q_STRIDE; d++)
                    require_close("qwen35 q rope gate untouched", out[d], in[d], 0.0f);
            }
            for (uint32_t h = 0; h < N_HEAD_KV; h++) {
                const float *in =
                    k_in + ((uint64_t)t * N_HEAD_KV + h) * KV_STRIDE;
                const float *out =
                    k_act + ((uint64_t)t * N_HEAD_KV + h) * KV_STRIDE;
                for (uint32_t d = N_ROT; d < HEAD_DIM; d++)
                    require_close("qwen35 k rope tail untouched", out[d], in[d], 0.0f);
            }
        }
        ds4_gpu_tensor_free(gk);
        ds4_gpu_tensor_free(gq);
        free(k_act); free(q_act); free(k_exp); free(q_exp);
        free(k_in); free(q_in);
    }

    /* ---------------------------------------------------------------- (a) */
    /* GQA FlashAttention: 16 query heads over 2 KV heads. */
    {
        const uint64_t q_count = (uint64_t)N_TOKENS * N_HEAD * HEAD_DIM;
        const uint64_t kv_count = (uint64_t)CACHE_CAP * N_HEAD_KV * HEAD_DIM;
        float *q = malloc(q_count * sizeof(float));
        float *k = calloc(kv_count, sizeof(float));
        float *v = calloc(kv_count, sizeof(float));
        float *expect = malloc(q_count * sizeof(float));
        float *actual = malloc(q_count * sizeof(float));
        require_ok(q && k && v && expect && actual, "gqa host allocation");
        for (uint64_t i = 0; i < q_count; i++)
            q[i] = synth((uint32_t)i, 41u, 0.0f);
        /* Query heads within a group are distinct, and the two KV heads are
         * distinct, so a broken grouping changes the output. */
        for (uint32_t t = 0; t < N_TOKENS; t++) {
            for (uint32_t h = 0; h < N_HEAD; h++) {
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    q[((uint64_t)t * N_HEAD + h) * HEAD_DIM + d] =
                        synth(h * HEAD_DIM + d + 1u, 43u, 0.05f) +
                        0.01f * (float)t;
                }
            }
            for (uint32_t h = 0; h < N_HEAD_KV; h++) {
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    const uint64_t idx =
                        ((uint64_t)t * N_HEAD_KV + h) * HEAD_DIM + d;
                    k[idx] = synth(h * HEAD_DIM + d + 7u, 47u, 0.05f);
                    v[idx] = synth(h * HEAD_DIM + d + 13u, 53u, 0.05f);
                }
            }
        }
        float group_delta = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM; d++)
            group_delta += fabsf(k[d] - k[HEAD_DIM + d]);
        require_ok(group_delta > 1.0f, "the two KV groups differ");

        reference_gqa(q, k, v, expect, N_TOKENS, N_HEAD, N_HEAD_KV, HEAD_DIM,
                      GROUP);

        ds4_gpu_tensor *gq =
            ds4_gpu_tensor_alloc(q_count * sizeof(float));
        ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
        ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(kv_count * sizeof(float));
        ds4_gpu_tensor *go =
            ds4_gpu_tensor_alloc(q_count * sizeof(float));
        require_ok(gq && gk && gv && go, "gqa tensor allocation");
        require_ok(ds4_gpu_tensor_write(gq, 0, q, q_count * sizeof(float)),
                   "gqa q write");
        require_ok(ds4_gpu_tensor_write(gk, 0, k, kv_count * sizeof(float)),
                   "gqa k write");
        require_ok(ds4_gpu_tensor_write(gv, 0, v, kv_count * sizeof(float)),
                   "gqa v write");
        require_ok(ds4_gpu_qwen35_attention_flash_tensor(
                       go, gq, gk, gv, 0, N_TOKENS, N_TOKENS, CACHE_CAP,
                       N_HEAD, N_HEAD_KV, HEAD_DIM, HEAD_DIM, false),
                   "qwen35 GQA flash attention");
        require_ok(ds4_gpu_tensor_read(go, 0, actual, q_count * sizeof(float)),
                   "gqa read");
        fprintf(stderr, "qwen35 attn metal: GQA max diff %.3g (16 q / 2 kv)\n",
                max_abs_diff(actual, expect, q_count));
        for (uint64_t i = 0; i < q_count; i++)
            require_close("qwen35 GQA attention", actual[i], expect[i], 3e-3f);

        ds4_gpu_tensor_free(go);
        ds4_gpu_tensor_free(gv);
        ds4_gpu_tensor_free(gk);
        ds4_gpu_tensor_free(gq);
        free(actual); free(expect); free(v); free(k); free(q);
    }

    ds4_gpu_cleanup();
    munmap(model, MODEL_BYTES);
    puts("qwen35 full-attention Metal kernels: PASS");
    return 0;
}
