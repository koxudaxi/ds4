/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE)
 * gated-delta-net (linear attention) CPU layer core.
 *
 * The layer is exposed by a test hook that ds4.c compiles under
 * -DDS4_TEST_HOOKS.  The hook is not declared in ds4.h; the contract is
 * duplicated here, exactly as tests/test_qwen35_attn.c duplicates the
 * full-attention hook.  Synthetic weights live in one anonymous mmap and are
 * handed to the hook as explicit f32 pointers, so the test measures the layer
 * and not the quantizer.
 *
 * The reference below is a plain scalar implementation of the same math:
 * fused qkv/z projections, the per-value-head beta/alpha/gate, the depthwise
 * causal conv (kernel 4) plus silu, the q/k/v split, per-head L2 norm, the
 * head repeat, the decayed delta-rule recurrence, the gated RMSNorm and the
 * output projection.  Each of the four Task 1 behaviours also has an ablation
 * assertion that removing it changes the output, so a silently-dropped conv,
 * decay, sk subtraction or z gate cannot pass.
 *
 * Task 2 adds the spec's remaining edge cases: a two-token recurrence, the
 * beta=0 (write suppressed, state only decays) and gate=0 (g=exp(0)=1, no
 * decay) branches, and a four-token case that pins the depthwise conv's
 * last-3-positions window.  The scalar reference gains force_beta_zero and
 * force_gate_zero so those branches are compared against an independent path,
 * and the state-equivalence case is the one a planted state reset breaks.
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-gdn
 *   ./external/ds4/tests/test_qwen35_gdn
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"

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
    N_TOKENS    = 3,
    MODEL_BYTES = 256 * 1024 * 1024,
    /* Per-layer state: the recurrent state (one [D_V, D_K] matrix per value
     * head) followed by the conv's last CONV_K-1 input rows. */
    REC_SIZE    = N_V * D_V * D_K,           /* 524288 */
    CONV_STATE  = (CONV_K - 1) * CONV_DIM,   /* 24576  */
    STATE_FLOATS = REC_SIZE + CONV_STATE,
};

/* Must match the struct ds4.c defines for the hook. */
typedef struct {
    const float *gdn_qkv;     /* [CONV_DIM][N_EMBD] */
    const float *gdn_conv1d;  /* [CONV_DIM][CONV_K] channel-major */
    const float *gdn_alpha;   /* [N_V][N_EMBD] */
    const float *gdn_beta;    /* [N_V][N_EMBD] */
    const float *gdn_a_log;   /* [N_V] */
    const float *gdn_dt_bias; /* [N_V] */
    const float *gdn_norm;    /* [D_V] */
    const float *gdn_out;     /* [N_EMBD][D_INNER] */
    const float *gdn_z;       /* [D_INNER][N_EMBD] */
} ds4_test_qwen35_gdn_weights;

int ds4_test_qwen35_gdn_forward(const ds4_test_qwen35_gdn_weights *w,
                                const float *x, uint32_t n_tokens,
                                float *state, float *out, FILE *dump);

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

/* Which of the four Task 1 behaviours a reference run applies.  Each is
 * toggled separately so the test can show it changes the output. */
typedef struct {
    int apply_conv;      /* 0 skips the depthwise conv and its silu */
    int apply_decay;     /* 0 sets g = 1 (no decay) */
    int apply_sk;        /* 0 drops the s.k subtraction */
    int apply_zgate;     /* 0 drops the silu(z) gate */
    int force_beta_zero; /* 1 sets b = 0: no write, the state only decays */
    int force_gate_zero; /* 1 sets gate = 0, so g = exp(0) = 1 (no decay) */
} gdn_config;

/* Independent scalar reference for the whole layer core.  `state` is the
 * per-layer buffer (recurrent then conv state), updated in place. */
static void reference_forward(const ds4_test_qwen35_gdn_weights *w,
                              const gdn_config *cfg, const float *x,
                              uint32_t n_tokens, float *state, float *out) {
    float *rec  = state;             /* [N_V][D_V][D_K], head-major */
    float *conv = state + REC_SIZE;  /* [CONV_K-1][CONV_DIM] */

    float qkv[CONV_DIM];
    float convout[CONV_DIM];
    float z[D_INNER];
    float beta[N_V], alpha[N_V], gate[N_V];
    float qn[N_V * D_K], kn[N_V * D_K];
    float o[N_V * D_V], on[N_V * D_V];

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *xt = x + (uint64_t)t * N_EMBD;
        matvec1(qkv, w->gdn_qkv, N_EMBD, CONV_DIM, xt);
        matvec1(z, w->gdn_z, N_EMBD, D_INNER, xt);
        matvec1(beta, w->gdn_beta, N_EMBD, N_V, xt);
        matvec1(alpha, w->gdn_alpha, N_EMBD, N_V, xt);

        if (cfg->apply_conv) {
            for (uint32_t ch = 0; ch < CONV_DIM; ch++) {
                double acc = 0.0;
                for (uint32_t kk = 0; kk < CONV_K; kk++) {
                    const float in = (kk < CONV_K - 1)
                                         ? conv[(uint64_t)kk * CONV_DIM + ch]
                                         : qkv[ch];
                    acc += (double)in * w->gdn_conv1d[(uint64_t)ch * CONV_K + kk];
                }
                convout[ch] = silu1((float)acc);
            }
            for (uint32_t kk = 0; kk + 1 < CONV_K - 1; kk++)
                memcpy(conv + (uint64_t)kk * CONV_DIM,
                       conv + (uint64_t)(kk + 1) * CONV_DIM,
                       CONV_DIM * sizeof(float));
            memcpy(conv + (uint64_t)(CONV_K - 2) * CONV_DIM, qkv,
                   CONV_DIM * sizeof(float));
        } else {
            memcpy(convout, qkv, sizeof(convout));
        }

        /* Per-head L2 norm of q and k, then repeat to the value-head count
         * and scale q by 1/sqrt(D_K).  The v half needs no repeat. */
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
            const float a = w->gdn_a_log[vh];
            gate[vh] = cfg->force_gate_zero
                           ? 0.0f
                           : a * softplus1(alpha[vh] + w->gdn_dt_bias[vh]);
            const float g = cfg->apply_decay ? expf(gate[vh]) : 1.0f;
            const float b = cfg->force_beta_zero ? 0.0f : sigmoid1(beta[vh]);

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
            for (uint32_t dv = 0; dv < D_V; dv++) {
                const float target = cfg->apply_sk
                                         ? vraw[(uint64_t)vh * D_V + dv] - sk[dv]
                                         : vraw[(uint64_t)vh * D_V + dv];
                dd[dv] = target * b;
            }
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
                float val = o[(uint64_t)vh * D_V + dv] * inv * w->gdn_norm[dv];
                if (cfg->apply_zgate)
                    val *= silu1(z[(uint64_t)vh * D_V + dv]);
                on[(uint64_t)vh * D_V + dv] = val;
            }
        }

        matvec1(out + (uint64_t)t * N_EMBD, w->gdn_out, D_INNER, N_EMBD, on);
    }
}

static void run_hook(const ds4_test_qwen35_gdn_weights *w, const float *x,
                     uint32_t n_tokens, float *state, float *out) {
    require_ok(ds4_test_qwen35_gdn_forward(w, x, n_tokens, state, out, NULL) == 0,
               "qwen35 gdn forward");
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

    float *wqkv = (float *)model;
    float *wconv = wqkv + (uint64_t)CONV_DIM * N_EMBD;
    float *walpha = wconv + (uint64_t)CONV_DIM * CONV_K;
    float *wbeta = walpha + (uint64_t)N_V * N_EMBD;
    float *walog = wbeta + (uint64_t)N_V * N_EMBD;
    float *wdt = walog + N_V;
    float *wnorm = wdt + N_V;
    float *wout = wnorm + D_V;
    float *wz = wout + (uint64_t)N_EMBD * D_INNER;
    /* Edge-case weight variants: a beta projection far negative enough that
     * sigmoid underflows to exactly 0, and an all-zero ssm_a (so gate = 0). */
    float *wbeta_zero = wz + (uint64_t)D_INNER * N_EMBD;
    float *wzero_log = wbeta_zero + (uint64_t)N_V * N_EMBD;
    require_ok((uint8_t *)(wz + (uint64_t)D_INNER * N_EMBD) <=
                   model + MODEL_BYTES,
               "synthetic weights fit the mapping");
    require_ok((uint8_t *)(wzero_log + N_V) <= model + MODEL_BYTES,
               "edge-case weights fit the mapping");

    for (uint32_t o = 0; o < CONV_DIM; o++)
        for (uint32_t i = 0; i < N_EMBD; i++)
            wqkv[(uint64_t)o * N_EMBD + i] = synth(o, i, 0.02f);
    for (uint32_t ch = 0; ch < CONV_DIM; ch++)
        for (uint32_t kk = 0; kk < CONV_K; kk++)
            wconv[(uint64_t)ch * CONV_K + kk] = synth(ch + 3u, kk, 0.35f);
    for (uint32_t o = 0; o < N_V; o++)
        for (uint32_t i = 0; i < N_EMBD; i++) {
            walpha[(uint64_t)o * N_EMBD + i] = synth(o + 11u, i, 0.01f);
            wbeta[(uint64_t)o * N_EMBD + i] = synth(o + 29u, i, 0.01f);
        }
    for (uint32_t h = 0; h < N_V; h++) {
        /* ssm_a is stored already folded as -exp(A_log), so every value is
         * negative; the magnitude sets how fast the recurrent state decays. */
        walog[h] = -(0.4f + 0.1f * (float)(h % 5u));
        wdt[h] = 0.2f * (float)((h % 3u)) - 0.2f;
    }
    for (uint32_t d = 0; d < D_V; d++)
        wnorm[d] = 0.75f + 0.0004f * (float)(d % 251u);
    for (uint32_t o = 0; o < N_EMBD; o++)
        for (uint32_t j = 0; j < D_INNER; j++)
            wout[(uint64_t)o * D_INNER + j] = synth(o + 101u, j, 0.02f);
    for (uint32_t o = 0; o < D_INNER; o++)
        for (uint32_t i = 0; i < N_EMBD; i++)
            wz[(uint64_t)o * N_EMBD + i] = synth(o + 211u, i, 0.02f);
    for (uint32_t o = 0; o < N_V; o++)
        for (uint32_t i = 0; i < N_EMBD; i++)
            wbeta_zero[(uint64_t)o * N_EMBD + i] = -1000.0f;
    for (uint32_t h = 0; h < N_V; h++) wzero_log[h] = 0.0f;

    ds4_test_qwen35_gdn_weights w = {
        .gdn_qkv = wqkv,
        .gdn_conv1d = wconv,
        .gdn_alpha = walpha,
        .gdn_beta = wbeta,
        .gdn_a_log = walog,
        .gdn_dt_bias = wdt,
        .gdn_norm = wnorm,
        .gdn_out = wout,
        .gdn_z = wz,
    };

    /* Nearly identical tokens ("identity input"): the key/value vectors stay
     * aligned across positions so the delta-rule state and its decay leave a
     * large, unmistakable signature from the second token on. */
    float x[N_TOKENS * N_EMBD];
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            x[(uint64_t)t * N_EMBD + d] =
                synth(d + 1u, 17u, 0.5f) + 0.01f * (float)t;

    /* All-ones input for the beta=0 case: with the -1000 beta rows the
     * projection is -1000*n_embd, so sigmoid underflows to exactly 0. */
    float xones[N_EMBD];
    for (uint32_t d = 0; d < N_EMBD; d++) xones[d] = 1.0f;

    const gdn_config all = {
        .apply_conv = 1,
        .apply_decay = 1,
        .apply_sk = 1,
        .apply_zgate = 1,
    };

    /* Single-token case, against the full reference. */
    {
        static float st_ref[STATE_FLOATS] = {0};
        static float st_hook[STATE_FLOATS] = {0};
        float expect[N_EMBD], actual[N_EMBD];
        reference_forward(&w, &all, x, 1, st_ref, expect);
        run_hook(&w, x, 1, st_hook, actual);
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 gdn single token", actual[d], expect[d],
                          1e-5f);
        require_ok(max_abs_diff(st_ref, st_hook, STATE_FLOATS) < 1e-6f,
                   "qwen35 gdn single-token state matches");
    }

    /* Two-token recurrence, against the full reference. */
    {
        static float st2_ref[STATE_FLOATS] = {0};
        static float st2_hook[STATE_FLOATS] = {0};
        float expect2[2 * N_EMBD], actual2[2 * N_EMBD];
        reference_forward(&w, &all, x, 2, st2_ref, expect2);
        run_hook(&w, x, 2, st2_hook, actual2);
        for (uint32_t d = 0; d < 2 * N_EMBD; d++)
            require_close("qwen35 gdn two tokens", actual2[d], expect2[d],
                          1e-5f);
    }

    /* Three-token case, against the full reference. */
    static float st_ref[STATE_FLOATS] = {0};
    static float st_hook[STATE_FLOATS] = {0};
    float expect[N_TOKENS * N_EMBD], actual[N_TOKENS * N_EMBD];
    reference_forward(&w, &all, x, N_TOKENS, st_ref, expect);
    run_hook(&w, x, N_TOKENS, st_hook, actual);
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 gdn three tokens",
                          actual[(uint64_t)t * N_EMBD + d],
                          expect[(uint64_t)t * N_EMBD + d], 1e-5f);

    /* State equivalence: token-by-token must equal the batched call, for both
     * the output and the final state.  This pins the conv carry-over and the
     * recurrent state layout. */
    {
        static float st_split[STATE_FLOATS] = {0};
        float out_split[N_TOKENS * N_EMBD];
        run_hook(&w, x, 1, st_split, out_split);
        run_hook(&w, x + N_EMBD, N_TOKENS - 1, st_split, out_split + N_EMBD);
        for (uint32_t t = 0; t < N_TOKENS; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                require_close("qwen35 gdn split call",
                              out_split[(uint64_t)t * N_EMBD + d],
                              actual[(uint64_t)t * N_EMBD + d], 0.0f);
        require_ok(max_abs_diff(st_split, st_hook, STATE_FLOATS) == 0.0f,
                   "qwen35 gdn split state is identical");
    }

    /* A non-zero state seeded by one normal token, used by the beta=0 and
     * gate=0 cases so the decay (and its absence) is observable. */
    static float st_seed[STATE_FLOATS] = {0};
    float st_seed_out[N_EMBD];
    reference_forward(&w, &all, x, 1, st_seed, st_seed_out);

    /* beta = 0: the delta-rule write is suppressed, so a non-zero state can
     * only decay.  The hook is driven to beta == 0 by the -1000 projection
     * (sigmoid underflows to exactly zero); the reference takes the same
     * branch via force_beta_zero. */
    {
        gdn_config no_beta = all;
        no_beta.force_beta_zero = 1;

        ds4_test_qwen35_gdn_weights wb = w;
        wb.gdn_beta = wbeta_zero;

        static float st_refb[STATE_FLOATS], st_hookb[STATE_FLOATS];
        memcpy(st_refb, st_seed, sizeof(st_seed));
        memcpy(st_hookb, st_seed, sizeof(st_seed));
        float expect_b[N_EMBD], actual_b[N_EMBD];
        reference_forward(&w, &no_beta, xones, 1, st_refb, expect_b);
        run_hook(&wb, xones, 1, st_hookb, actual_b);
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 gdn beta=0", actual_b[d], expect_b[d],
                          1e-5f);
        require_ok(max_abs_diff(st_refb, st_hookb, STATE_FLOATS) < 1e-6f,
                   "qwen35 gdn beta=0 state matches");

        /* The suppression must matter: with beta live the write moves it. */
        static float st_live[STATE_FLOATS];
        memcpy(st_live, st_seed, sizeof(st_seed));
        float live_out[N_EMBD];
        reference_forward(&w, &all, xones, 1, st_live, live_out);
        require_ok(max_abs_diff(expect_b, live_out, N_EMBD) > 1e-4f,
                   "beta=0 suppresses the delta write");
    }

    /* gate = 0: the log-decay is zero, so g = exp(0) = 1 and the state does
     * not decay.  The hook is driven by an all-zero ssm_a; the reference takes
     * force_gate_zero. */
    {
        gdn_config no_gate = all;
        no_gate.force_gate_zero = 1;

        ds4_test_qwen35_gdn_weights wg = w;
        wg.gdn_a_log = wzero_log;

        static float st_refg[STATE_FLOATS], st_hookg[STATE_FLOATS];
        memcpy(st_refg, st_seed, sizeof(st_seed));
        memcpy(st_hookg, st_seed, sizeof(st_seed));
        float expect_g[N_EMBD], actual_g[N_EMBD];
        reference_forward(&w, &no_gate, xones, 1, st_refg, expect_g);
        run_hook(&wg, xones, 1, st_hookg, actual_g);
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 gdn gate=0", actual_g[d], expect_g[d],
                          1e-5f);
        require_ok(max_abs_diff(st_refg, st_hookg, STATE_FLOATS) < 1e-6f,
                   "qwen35 gdn gate=0 state matches");

        /* The zero gate must matter: with decay live the state shrinks. */
        static float st_decay[STATE_FLOATS];
        memcpy(st_decay, st_seed, sizeof(st_seed));
        float decay_out[N_EMBD];
        reference_forward(&w, &all, xones, 1, st_decay, decay_out);
        require_ok(max_abs_diff(expect_g, decay_out, N_EMBD) > 1e-4f,
                   "gate=0 disables decay");
    }

    /* Conv window: the depthwise conv is a 4-tap causal filter, so after four
     * tokens its state holds the raw qkv projections of tokens 1,2,3 (oldest
     * first) and token 0 has fallen out.  Recompute those rows and require the
     * stored state to match -- a 3-tap window or the wrong order cannot pass.
     * Token 0 is also three positions back from token 3's output, so changing
     * it must move that output. */
    {
        enum { CONV_TOKENS = 4 };
        float xc[CONV_TOKENS * N_EMBD];
        for (uint32_t t = 0; t < CONV_TOKENS; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                xc[(uint64_t)t * N_EMBD + d] =
                    synth(d + 1u, 31u, 0.5f) + 0.05f * (float)t;

        static float st_conv[STATE_FLOATS] = {0};
        float out_conv[CONV_TOKENS * N_EMBD];
        run_hook(&w, xc, CONV_TOKENS, st_conv, out_conv);

        /* The four-token hook output still matches the full reference. */
        static float st_conv_ref[STATE_FLOATS] = {0};
        float out_conv_ref[CONV_TOKENS * N_EMBD];
        reference_forward(&w, &all, xc, CONV_TOKENS, st_conv_ref,
                          out_conv_ref);
        for (uint32_t d = 0; d < CONV_TOKENS * N_EMBD; d++)
            require_close("qwen35 gdn conv four tokens", out_conv[d],
                          out_conv_ref[d], 1e-5f);

        /* The last three raw qkv rows, oldest first. */
        for (uint32_t t = 1; t < CONV_TOKENS; t++) {
            float qrow[CONV_DIM];
            matvec1(qrow, wqkv, N_EMBD, CONV_DIM, xc + (uint64_t)t * N_EMBD);
            const float *stored =
                st_conv + REC_SIZE + (uint64_t)(t - 1) * CONV_DIM;
            for (uint32_t ch = 0; ch < CONV_DIM; ch++)
                require_close("qwen35 gdn conv window row", stored[ch],
                              qrow[ch], 1e-6f);
        }

        float xc_alt[CONV_TOKENS * N_EMBD];
        memcpy(xc_alt, xc, sizeof(xc));
        for (uint32_t d = 0; d < N_EMBD; d++) xc_alt[d] = 0.0f;
        static float st_alt[STATE_FLOATS] = {0};
        float out_alt[CONV_TOKENS * N_EMBD];
        run_hook(&w, xc_alt, CONV_TOKENS, st_alt, out_alt);
        require_ok(max_abs_diff(out_conv + (uint64_t)3 * N_EMBD,
                                out_alt + (uint64_t)3 * N_EMBD, N_EMBD) >
                       1e-4f,
                   "conv output depends on the token 3 positions back");
    }

    /* Ablation sensitivity: the four behaviours must all change the output.
     * A silently-dropped conv, decay, sk subtraction or z gate cannot pass. */
    {
        gdn_config abl[4] = { all, all, all, all };
        const char *names[4] = {
            "conv changes the output",
            "decay changes the output",
            "s.k subtraction changes the output",
            "silu(z) gate changes the output",
        };
        abl[0].apply_conv = 0;
        abl[1].apply_decay = 0;
        abl[2].apply_sk = 0;
        abl[3].apply_zgate = 0;
        for (int i = 0; i < 4; i++) {
            static float st_abl[STATE_FLOATS];
            memset(st_abl, 0, sizeof(st_abl));
            float abl_out[N_TOKENS * N_EMBD];
            reference_forward(&w, &abl[i], x, N_TOKENS, st_abl, abl_out);
            require_ok(max_abs_diff(actual, abl_out, N_TOKENS * N_EMBD) > 1e-4f,
                       names[i]);
        }
    }

    /* Folded-coefficient sensitivity: ssm_a is stored as -exp(A_log), i.e.
     * already the folded decay coefficient.  Feeding the hook the raw log
     * coefficient (-exp of the stored value) must change the output, so a
     * re-applied exp in the hook cannot pass silently. */
    {
        float walog_refolded[N_V];
        for (uint32_t h = 0; h < N_V; h++)
            walog_refolded[h] = -expf(walog[h]);
        ds4_test_qwen35_gdn_weights wr = w;
        wr.gdn_a_log = walog_refolded;
        static float st_refold[STATE_FLOATS] = {0};
        float out_refold[N_TOKENS * N_EMBD];
        run_hook(&wr, x, N_TOKENS, st_refold, out_refold);
        require_ok(max_abs_diff(actual, out_refold, N_TOKENS * N_EMBD) > 1e-4f,
                   "re-folding ssm_a changes the output");
    }

    /* The hook must reject a null weight pointer and a zero-token call. */
    require_ok(ds4_test_qwen35_gdn_forward(NULL, x, 1, st_hook, actual, NULL) != 0,
               "gdn forward rejects a null weight struct");
    require_ok(ds4_test_qwen35_gdn_forward(&w, x, 0, st_hook, actual, NULL) != 0,
               "gdn forward rejects zero tokens");

    munmap(model, MODEL_BYTES);
    puts("qwen35 gated-delta-net layer core: PASS");
    return 0;
}
