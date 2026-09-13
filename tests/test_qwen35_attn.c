/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE)
 * full-attention CPU layer core.
 *
 * The layer is exposed by a test hook that ds4.c compiles under
 * -DDS4_TEST_HOOKS.  The hook is not declared in ds4.h; the contract is
 * duplicated here, exactly as tests/test_qwen35_tokenizer.c duplicates the
 * other qwen35 hooks.  Synthetic weights live in one anonymous mmap and are
 * handed to the hook as explicit f32 pointers, so the test measures the layer
 * and not the quantizer.
 *
 * The reference below is a plain scalar implementation of the same math:
 * pre-attention RMSNorm, q/k/v projections (attn_q is double width: query then
 * sigmoid output gate), per-head q/k RMSNorm, plain partial RoPE over the tail
 * n_rot of each head, GQA grouping (n_head query heads share n_head_kv KV heads
 * by h / (n_head / n_head_kv)), a causal softmax over key_pos <= query_pos, the
 * sigmoid gate applied to the attention output, the output projection, the
 * attention residual and the post-attention RMSNorm.  Each of the three Task 2
 * pieces also has a sensitivity assertion that removing it changes the output,
 * so a silently-dropped norm, rope or gate cannot pass.
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-attn
 *   ./external/ds4/tests/test_qwen35_attn
 */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "ds4.h"

/* Ornith-1.5-35B-A3B (DS4_SHAPE_ORNITH15) full-attention geometry. */
enum {
    N_EMBD      = 2048,
    N_HEAD      = 16,
    N_HEAD_KV   = 2,
    HEAD_DIM    = 256,
    N_ROT       = 64,
    Q_DIM       = N_HEAD * HEAD_DIM,      /* 4096 */
    Q_GATE_DIM  = 2 * Q_DIM,              /* 8192 */
    KV_DIM      = N_HEAD_KV * HEAD_DIM,   /* 512 */
    GROUP       = N_HEAD / N_HEAD_KV,     /* 8   */
    N_TOKENS    = 3,
    MODEL_BYTES = 128 * 1024 * 1024,
};

/* Must match the struct ds4.c defines for the hook. */
typedef struct {
    const float *attn_q;       /* [Q_GATE_DIM][N_EMBD] */
    const float *attn_k;       /* [KV_DIM][N_EMBD] */
    const float *attn_v;       /* [KV_DIM][N_EMBD] */
    const float *attn_output;  /* [N_EMBD][Q_DIM] */
    const float *attn_norm;    /* [N_EMBD] pre-attention */
    const float *ffn_norm;     /* [N_EMBD] post-attention */
    const float *attn_q_norm;  /* [HEAD_DIM] per-head q RMSNorm */
    const float *attn_k_norm;  /* [HEAD_DIM] per-head k RMSNorm */
    const float *x;            /* [n_tokens][N_EMBD] */
    float       *out;          /* [n_tokens][N_EMBD] */
    uint32_t     n_tokens;
    uint32_t     il;
    uint32_t     n_rot;        /* partial-RoPE tail, 0 disables */
    uint32_t     pos0;         /* position of the first token */
    float        rope_freq_base;
} ds4_test_qwen35_attn_args;

int ds4_test_qwen35_attn_forward(const ds4_test_qwen35_attn_args *args);

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

/* Deterministic, zero-mean-ish synthetic values.  A multiplicative hash keeps
 * the weights varied enough that the two KV groups disagree and that a broken
 * GQA grouping changes the output by far more than the tolerance. */
static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

static float rms_norm_one(const float *x, const float *weight, uint64_t n,
                          float *out) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + 1e-6f);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
    return scale;
}

/* Partial RoPE over the first N_ROT dims of each head, Neox pairing: pair
 * (p, p + n_rot/2) rotated by pos * freq_base^(-2p/n_rot).  This is what
 * llama.cpp's ggml_rope_multi does for qwen35moe's IMROPE mode at a text-only
 * position.  `stride` is the per-head stride (2*HEAD_DIM for the interleaved
 * query+gate projection, HEAD_DIM for K). */
static void rope_neox_front_one(float *x, uint32_t n_head, uint32_t stride,
                                uint32_t n_rot, uint32_t pos, float freq_base) {
    if (n_rot == 0) return;
    const uint32_t half = n_rot / 2u;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * stride;
        float theta = (float)pos;
        for (uint32_t p = 0; p < half; p++) {
            const float c = cosf(theta);
            const float s = sinf(theta);
            const float x0 = head[p];
            const float x1 = head[p + half];
            head[p]        = x0 * c - x1 * s;
            head[p + half] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
    }
}

/* Which of the three Task 2 behaviours a reference run applies.  Each is
 * toggled separately so the test can show it changes the output. */
typedef struct {
    const float *q_norm;      /* [HEAD_DIM] per-head q RMSNorm weight */
    const float *k_norm;      /* [HEAD_DIM] per-head k RMSNorm weight */
    uint32_t     n_rot;       /* 0 disables RoPE */
    uint32_t     pos0;
    float        rope_freq_base;
    int          apply_gate;  /* 0 skips the sigmoid output gate */
} layer_config;

/* Independent scalar reference for the whole layer core. */
static void reference_layer(const float *wq, const float *wk, const float *wv,
                            const float *wo, const float *norm_pre,
                            const float *norm_post, const layer_config *cfg,
                            const float *x, uint32_t n_tokens, float *out) {
    static float h[N_TOKENS * N_EMBD];
    static float q[N_TOKENS * Q_GATE_DIM];
    static float k[N_TOKENS * KV_DIM];
    static float v[N_TOKENS * KV_DIM];
    static float context[N_TOKENS * Q_DIM];
    static float proj[N_TOKENS * N_EMBD];
    static float resid[N_TOKENS * N_EMBD];

    for (uint32_t t = 0; t < n_tokens; t++) {
        rms_norm_one(x + (uint64_t)t * N_EMBD, norm_pre, N_EMBD,
                     h + (uint64_t)t * N_EMBD);
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ht = h + (uint64_t)t * N_EMBD;
        float *qt = q + (uint64_t)t * Q_GATE_DIM;
        float *kt = k + (uint64_t)t * KV_DIM;
        float *vt = v + (uint64_t)t * KV_DIM;
        for (uint32_t o = 0; o < Q_GATE_DIM; o++) {
            double acc = 0.0;
            const float *row = wq + (uint64_t)o * N_EMBD;
            for (uint32_t i = 0; i < N_EMBD; i++) acc += (double)row[i] * ht[i];
            qt[o] = (float)acc;
        }
        for (uint32_t o = 0; o < KV_DIM; o++) {
            double acck = 0.0, accv = 0.0;
            const float *rk = wk + (uint64_t)o * N_EMBD;
            const float *rv = wv + (uint64_t)o * N_EMBD;
            for (uint32_t i = 0; i < N_EMBD; i++) {
                acck += (double)rk[i] * ht[i];
                accv += (double)rv[i] * ht[i];
            }
            kt[o] = (float)acck;
            vt[o] = (float)accv;
        }
    }

    /* Query and gate are interleaved per head: head hd owns query at
     * hd*2*HEAD_DIM and gate immediately after.  Normalise and rotate the
     * query block in place; K is contiguous per head. */
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *qt = q + (uint64_t)t * Q_GATE_DIM;
        for (uint32_t hd = 0; hd < N_HEAD; hd++)
            rms_norm_one(qt + (uint64_t)hd * 2 * HEAD_DIM, cfg->q_norm, HEAD_DIM,
                         qt + (uint64_t)hd * 2 * HEAD_DIM);
        float *kt = k + (uint64_t)t * KV_DIM;
        for (uint32_t hd = 0; hd < N_HEAD_KV; hd++)
            rms_norm_one(kt + (uint64_t)hd * HEAD_DIM, cfg->k_norm, HEAD_DIM,
                         kt + (uint64_t)hd * HEAD_DIM);
        if (cfg->n_rot != 0) {
            rope_neox_front_one(qt, N_HEAD, 2 * HEAD_DIM, cfg->n_rot,
                                cfg->pos0 + t, cfg->rope_freq_base);
            rope_neox_front_one(kt, N_HEAD_KV, HEAD_DIM, cfg->n_rot,
                                cfg->pos0 + t, cfg->rope_freq_base);
        }
    }

    const float scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t hd = 0; hd < N_HEAD; hd++) {
            const uint32_t kv_head = hd / GROUP;
            const float *qh = q + (uint64_t)t * Q_GATE_DIM + (uint64_t)hd * 2 * HEAD_DIM;
            float *ctx = context + (uint64_t)t * Q_DIM + (uint64_t)hd * HEAD_DIM;
            float scores[N_TOKENS];
            float max_score = -FLT_MAX;
            for (uint32_t s = 0; s <= t; s++) {
                const float *kh =
                    k + (uint64_t)s * KV_DIM + (uint64_t)kv_head * HEAD_DIM;
                double dot = 0.0;
                for (uint32_t d = 0; d < HEAD_DIM; d++) {
                    dot += (double)qh[d] * kh[d];
                }
                scores[s] = (float)dot * scale;
                if (scores[s] > max_score) max_score = scores[s];
            }
            double denom = 0.0;
            for (uint32_t s = 0; s <= t; s++) {
                scores[s] = expf(scores[s] - max_score);
                denom += scores[s];
            }
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                double acc = 0.0;
                for (uint32_t s = 0; s <= t; s++) {
                    const float *vh =
                        v + (uint64_t)s * KV_DIM + (uint64_t)kv_head * HEAD_DIM;
                    acc += ((double)scores[s] / denom) * vh[d];
                }
                ctx[d] = (float)acc;
            }
        }
    }

    /* Sigmoid output gate: the second half of attn_q multiplies the attention
     * output elementwise before the output projection. */
    if (cfg->apply_gate) {
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float *qt = q + (uint64_t)t * Q_GATE_DIM;
            float *ct = context + (uint64_t)t * Q_DIM;
            for (uint32_t hd = 0; hd < N_HEAD; hd++) {
                const float *gate =
                    qt + (uint64_t)hd * 2 * HEAD_DIM + HEAD_DIM;
                float *ch = ct + (uint64_t)hd * HEAD_DIM;
                for (uint32_t d = 0; d < HEAD_DIM; d++)
                    ch[d] *= 1.0f / (1.0f + expf(-gate[d]));
            }
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ct = context + (uint64_t)t * Q_DIM;
        for (uint32_t o = 0; o < N_EMBD; o++) {
            double acc = 0.0;
            const float *row = wo + (uint64_t)o * Q_DIM;
            for (uint32_t j = 0; j < Q_DIM; j++) acc += (double)row[j] * ct[j];
            proj[(uint64_t)t * N_EMBD + o] = (float)acc;
        }
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t d = 0; d < N_EMBD; d++) {
            const uint64_t idx = (uint64_t)t * N_EMBD + d;
            resid[idx] = x[idx] + proj[idx];
        }
        rms_norm_one(resid + (uint64_t)t * N_EMBD, norm_post, N_EMBD,
                     out + (uint64_t)t * N_EMBD);
    }
}

/* Drive the hook with cfg through the test's local forward declaration. */
static void run_hook(const float *wq, const float *wk, const float *wv,
                     const float *wo, const float *norm_pre,
                     const float *norm_post, const layer_config *cfg,
                     const float *x, uint32_t n_tokens, uint32_t il,
                     float *out) {
    ds4_test_qwen35_attn_args args = {
        .attn_q = wq,
        .attn_k = wk,
        .attn_v = wv,
        .attn_output = wo,
        .attn_norm = norm_pre,
        .ffn_norm = norm_post,
        .attn_q_norm = cfg->q_norm,
        .attn_k_norm = cfg->k_norm,
        .x = x,
        .out = out,
        .n_tokens = n_tokens,
        .il = il,
        .n_rot = cfg->n_rot,
        .pos0 = cfg->pos0,
        .rope_freq_base = cfg->rope_freq_base,
    };
    require_ok(ds4_test_qwen35_attn_forward(&args) == 0,
               "qwen35 attn forward");
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

    float *wq = (float *)model;
    float *wk = wq + (uint64_t)Q_GATE_DIM * N_EMBD;
    float *wv = wk + (uint64_t)KV_DIM * N_EMBD;
    float *wo = wv + (uint64_t)KV_DIM * N_EMBD;
    float *norm_pre = wo + (uint64_t)N_EMBD * Q_DIM;
    float *norm_post = norm_pre + N_EMBD;
    float *q_norm = norm_post + N_EMBD;
    float *k_norm = q_norm + HEAD_DIM;
    require_ok((uint8_t *)(k_norm + HEAD_DIM) <= model + MODEL_BYTES,
               "synthetic weights fit the mapping");

    for (uint32_t o = 0; o < Q_GATE_DIM; o++)
        for (uint32_t i = 0; i < N_EMBD; i++)
            wq[(uint64_t)o * N_EMBD + i] = synth(o, i, 0.02f);
    for (uint32_t o = 0; o < KV_DIM; o++)
        for (uint32_t i = 0; i < N_EMBD; i++) {
            wk[(uint64_t)o * N_EMBD + i] = synth(o + 7u, i, 0.02f);
            wv[(uint64_t)o * N_EMBD + i] = synth(o + 23u, i, 0.02f);
        }
    for (uint32_t o = 0; o < N_EMBD; o++)
        for (uint32_t j = 0; j < Q_DIM; j++)
            wo[(uint64_t)o * Q_DIM + j] = synth(o + 101u, j, 0.02f);
    for (uint32_t d = 0; d < N_EMBD; d++) {
        norm_pre[d] = 0.9f + 0.0002f * (float)(d % 997u);
        norm_post[d] = 1.1f - 0.00015f * (float)(d % 991u);
    }
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        q_norm[d] = 0.8f + 0.0004f * (float)(d % 251u);
        k_norm[d] = 1.0f + 0.0003f * (float)(d % 199u);
    }

    float x[N_TOKENS * N_EMBD];
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            x[(uint64_t)t * N_EMBD + d] =
                synth(d + 1u, t + 17u, 0.6f) + 0.01f * (float)t;

    const layer_config all = {
        .q_norm = q_norm,
        .k_norm = k_norm,
        .n_rot = N_ROT,
        .pos0 = 3,
        .rope_freq_base = 10000000.0f,
        .apply_gate = 1,
    };

    /* Case (a): per-head q/k RMSNorm, isolated (no RoPE), against the
     * reference.  Run first so a broken norm fails this named case. */
    layer_config norm_cfg = all;
    norm_cfg.n_rot = 0;
    norm_cfg.pos0 = 0;
    float expected_norm[N_TOKENS * N_EMBD];
    reference_layer(wq, wk, wv, wo, norm_pre, norm_post, &norm_cfg, x, N_TOKENS,
                    expected_norm);
    float actual_norm[N_TOKENS * N_EMBD];
    run_hook(wq, wk, wv, wo, norm_pre, norm_post, &norm_cfg, x, N_TOKENS, 3,
             actual_norm);
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 per-head q/k norm",
                          actual_norm[(uint64_t)t * N_EMBD + d],
                          expected_norm[(uint64_t)t * N_EMBD + d], 1e-5f);

    /* The per-head norm weights must matter: unit weights change the output. */
    float unit_q[HEAD_DIM], unit_k[HEAD_DIM];
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        unit_q[d] = 1.0f;
        unit_k[d] = 1.0f;
    }
    layer_config unit_cfg = norm_cfg;
    unit_cfg.q_norm = unit_q;
    unit_cfg.k_norm = unit_k;
    float actual_unit[N_TOKENS * N_EMBD];
    run_hook(wq, wk, wv, wo, norm_pre, norm_post, &unit_cfg, x, N_TOKENS, 3,
             actual_unit);
    require_ok(max_abs_diff(actual_norm, actual_unit, N_TOKENS * N_EMBD) > 1e-4f,
               "per-head q/k norm changes the output");

    /* Cases (b)/(d): a multi-token causal batch with per-head norm, partial
     * RoPE and the sigmoid gate all active, against the full naive reference.
     * Run after the norm-only case so a broken RoPE or gate attributes here. */
    float expected[N_TOKENS * N_EMBD];
    reference_layer(wq, wk, wv, wo, norm_pre, norm_post, &all, x, N_TOKENS,
                    expected);
    float actual[N_TOKENS * N_EMBD];
    run_hook(wq, wk, wv, wo, norm_pre, norm_post, &all, x, N_TOKENS, 3, actual);
    for (uint32_t t = 0; t < N_TOKENS; t++)
        for (uint32_t d = 0; d < N_EMBD; d++)
            require_close("qwen35 attn layer (norm+rope+gate)",
                          actual[(uint64_t)t * N_EMBD + d],
                          expected[(uint64_t)t * N_EMBD + d], 1e-5f);

    /* The first two rows of a 3-token batch must equal a 1- and a 2-token run:
     * this pins causal masking (a later key must not influence an earlier
     * query) and the no-mixing single-token case, now with all three Task 2
     * behaviours active. */
    for (uint32_t n = 1; n <= 2; n++) {
        float prefix[N_TOKENS * N_EMBD];
        run_hook(wq, wk, wv, wo, norm_pre, norm_post, &all, x, n, 3, prefix);
        for (uint32_t t = 0; t < n; t++)
            for (uint32_t d = 0; d < N_EMBD; d++)
                require_close("qwen35 causal prefix",
                              prefix[(uint64_t)t * N_EMBD + d],
                              actual[(uint64_t)t * N_EMBD + d], 0.0f);
    }

    /* Case (b): partial RoPE.  The reference rotates only the tail N_ROT dims;
     * disabling n_rot must change the output, so a full-head or absent rotation
     * cannot pass. */
    layer_config norope_cfg = all;
    norope_cfg.n_rot = 0;
    norope_cfg.pos0 = 0;
    float actual_norope[N_TOKENS * N_EMBD];
    run_hook(wq, wk, wv, wo, norm_pre, norm_post, &norope_cfg, x, N_TOKENS, 3,
             actual_norope);
    require_ok(max_abs_diff(actual, actual_norope, N_TOKENS * N_EMBD) > 1e-4f,
               "partial RoPE changes the output");

    /* Case (c): the sigmoid output gate.  A reference that skips the gate must
     * differ from the gated output, so a zeroed or discarded gate cannot
     * pass. */
    layer_config nogate_cfg = all;
    nogate_cfg.apply_gate = 0;
    float expected_nogate[N_TOKENS * N_EMBD];
    reference_layer(wq, wk, wv, wo, norm_pre, norm_post, &nogate_cfg, x,
                    N_TOKENS, expected_nogate);
    require_ok(max_abs_diff(actual, expected_nogate, N_TOKENS * N_EMBD) > 1e-3f,
               "sigmoid output gate changes the output");

    /* The two KV groups must actually differ; otherwise the GQA test above is
     * vacuous.  Compare the kv=0 and kv=1 K slices of the first projection. */
    float group_delta = 0.0f;
    for (uint32_t d = 0; d < HEAD_DIM; d++) {
        group_delta += fabsf(wk[d] - wk[(uint64_t)HEAD_DIM + d]);
    }
    require_ok(group_delta > 1.0f, "the two KV groups differ");

    /* il=3 is a full-attention block; il=4 is a linear (gated delta net) block
     * per ds4_qwen35moe_layer_is_linear.  The hook must refuse the latter: it
     * has no attn_q/k/v and is not the layer this core models. */
    ds4_test_qwen35_attn_args bad = {
        .attn_q = wq,
        .attn_k = wk,
        .attn_v = wv,
        .attn_output = wo,
        .attn_norm = norm_pre,
        .ffn_norm = norm_post,
        .attn_q_norm = q_norm,
        .attn_k_norm = k_norm,
        .x = x,
        .out = actual,
        .n_tokens = N_TOKENS,
        .il = 4,
        .n_rot = N_ROT,
        .pos0 = 0,
        .rope_freq_base = 10000000.0f,
    };
    require_ok(ds4_test_qwen35_attn_forward(&bad) != 0,
               "attn forward rejects a linear layer");

    munmap(model, MODEL_BYTES);
    puts("qwen35 full-attention layer core: PASS");
    return 0;
}
