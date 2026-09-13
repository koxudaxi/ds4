/* Model-free test for the qwen35moe (Ornith 1.5 35B / Qwen3.5-MoE) Metal MoE
 * router kernel (kernel_qwen35_moe_route / ds4_gpu_qwen35_moe_route_tensor).
 *
 * The CPU hook qwen35_moe_route (ds4.c) is the golden reference: a max-
 * subtracted softmax over ALL experts in float32, then top-k selection by
 * probability with ties resolved to the LOWER expert index, then
 * renormalisation of the selected k weights.  The parallel kernel replaced the
 * old lane-0 selection sort, so this test pins the two things that a parallel
 * reduction can silently change: the tie-break order and the k == n path.
 *
 * Cases:
 *   - random 256 experts, k=8 (indices exact, weights within tolerance);
 *   - uniform 256 experts, k=8 (every probability is an exact tie: the top-8
 *     must be experts 0..7 in that order and every weight exactly 1/8);
 *   - a constructed top tie (two equal maxima): the lower index must win and
 *     come first;
 *   - k == n == 8 (all experts selected once, sorted by descending value,
 *     weights renormalised to sum 1);
 *   - a three-token batch, so the per-token threadgroup grid is exercised.
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-moe-route-metal
 *   ./external/ds4/tests/test_qwen35_moe_route_metal
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

enum { N_EXPERT = 256, N_USED = 8, N_TOKENS = 3 };

static void require_ok(int ok, const char *what) {
    if (!ok) {
        fprintf(stderr, "%s failed\n", what);
        exit(1);
    }
}

/* Deterministic, zero-mean-ish synthetic values, distinct for distinct (o,i). */
static float synth(uint32_t o, uint32_t i, float scale) {
    uint32_t v = (o * 2654435761u) ^ (i * 40503u) ^ 0x9e3779b9u;
    v ^= v >> 13;
    v *= 2246822519u;
    v ^= v >> 16;
    const int32_t s = (int32_t)((v >> 8) % 2001u) - 1000;
    return scale * (float)s / 1000.0f;
}

/* Plain scalar port of ds4.c's qwen35_moe_route.  Selection scans in index
 * order with a strict `>`, so an equal probability keeps the lower index. */
static uint32_t reference_route(const float *logits, uint32_t n, uint32_t k,
                                uint32_t *idx, float *w) {
    static float probs[512];
    static float sel[512];
    static bool used[512];
    if (k > n) k = n;

    float max_logit = logits[0];
    for (uint32_t i = 1; i < n; i++)
        if (logits[i] > max_logit) max_logit = logits[i];

    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        probs[i] = (float)exp((double)logits[i] - (double)max_logit);
        sum += probs[i];
    }
    for (uint32_t i = 0; i < n; i++) probs[i] = (float)(probs[i] / sum);

    for (uint32_t i = 0; i < n; i++) used[i] = false;
    for (uint32_t t = 0; t < k; t++) {
        uint32_t best = UINT32_MAX;
        for (uint32_t i = 0; i < n; i++) {
            if (used[i]) continue;
            if (best == UINT32_MAX || probs[i] > probs[best]) best = i;
        }
        used[best] = true;
        idx[t] = best;
        sel[t] = probs[best];
    }

    double wsum = 0.0;
    for (uint32_t t = 0; t < k; t++) wsum += sel[t];
    for (uint32_t t = 0; t < k; t++) w[t] = (float)(sel[t] / wsum);
    return k;
}

/* Run one batch through the GPU kernel and check it against the scalar
 * reference, which is recomputed here row by row. */
static void check_case(const char *what, const float *logits, uint32_t n_tokens,
                       uint32_t n_expert, uint32_t n_used, float scale,
                       float tol) {
    const uint64_t logits_bytes = (uint64_t)n_tokens * n_expert * sizeof(float);
    const uint64_t sel_bytes = (uint64_t)n_tokens * n_used * sizeof(int32_t);
    const uint64_t w_bytes = (uint64_t)n_tokens * n_used * sizeof(float);

    ds4_gpu_tensor *g_logits = ds4_gpu_tensor_alloc(logits_bytes);
    ds4_gpu_tensor *g_sel = ds4_gpu_tensor_alloc(sel_bytes);
    ds4_gpu_tensor *g_w = ds4_gpu_tensor_alloc(w_bytes);
    require_ok(g_logits && g_sel && g_w, "router tensor allocation");
    require_ok(ds4_gpu_tensor_write(g_logits, 0, logits, logits_bytes),
               "logits write");

    require_ok(ds4_gpu_qwen35_moe_route_tensor(g_sel, g_w, g_logits, n_expert,
                                               n_used, scale, n_tokens),
               what);

    static int32_t got_sel[64];
    static float got_w[64];
    require_ok(ds4_gpu_tensor_read(g_sel, 0, got_sel, sel_bytes),
               "selected read");
    require_ok(ds4_gpu_tensor_read(g_w, 0, got_w, w_bytes), "weights read");

    float max_wdiff = 0.0f;
    for (uint32_t r = 0; r < n_tokens; r++) {
        uint32_t exp_idx[32];
        float exp_w[32];
        const uint32_t k = reference_route(logits + (uint64_t)r * n_expert,
                                           n_expert, n_used, exp_idx, exp_w);
        require_ok(k == n_used, "reference returns k");
        for (uint32_t t = 0; t < k; t++) {
            const int32_t g = got_sel[(uint64_t)r * n_used + t];
            if ((uint32_t)g != exp_idx[t]) {
                fprintf(stderr,
                        "%s: row %u slot %u selected %d, expected %u\n",
                        what, r, t, g, exp_idx[t]);
                exit(1);
            }
            const float gw = got_w[(uint64_t)r * n_used + t];
            const float ew = exp_w[t] * scale;
            if (!isfinite(gw) || fabsf(gw - ew) > tol) {
                fprintf(stderr,
                        "%s: row %u slot %u weight %.9g, expected %.9g "
                        "(tolerance %.3g)\n",
                        what, r, t, gw, ew, tol);
                exit(1);
            }
            max_wdiff = fmaxf(max_wdiff, fabsf(gw - ew));
        }
    }

    fprintf(stderr, "%s: max weight diff %.3g (n=%u tokens)\n", what, max_wdiff,
            n_tokens);
    ds4_gpu_tensor_free(g_w);
    ds4_gpu_tensor_free(g_sel);
    ds4_gpu_tensor_free(g_logits);
}

int main(void) {
    require_ok(ds4_gpu_init(), "GPU initialization");

    /* 1. Random 256 experts, k=8, three tokens. */
    {
        static float logits[(uint64_t)N_TOKENS * N_EXPERT];
        for (uint32_t r = 0; r < N_TOKENS; r++)
            for (uint32_t e = 0; e < N_EXPERT; e++)
                logits[(uint64_t)r * N_EXPERT + e] = synth(r + 1u, e, 2.0f);
        check_case("qwen35 router random (256 experts, k=8)", logits, N_TOKENS,
                   N_EXPERT, N_USED, 1.0f, 1e-6f);
    }

    /* 2. Uniform logits are exact ties: top-8 must be 0..7 in order, and every
     *    weight exactly 1/8. */
    {
        static float logits[N_EXPERT];
        for (uint32_t e = 0; e < N_EXPERT; e++) logits[e] = 0.5f;
        check_case("qwen35 router uniform tie (256 experts, k=8)", logits, 1,
                   N_EXPERT, N_USED, 1.0f, 1e-6f);

        uint32_t idx[32];
        float w[32];
        ds4_gpu_tensor *g_logits = ds4_gpu_tensor_alloc(N_EXPERT * sizeof(float));
        ds4_gpu_tensor *g_sel = ds4_gpu_tensor_alloc(N_USED * sizeof(int32_t));
        ds4_gpu_tensor *g_w = ds4_gpu_tensor_alloc(N_USED * sizeof(float));
        require_ok(g_logits && g_sel && g_w, "tie tensor allocation");
        require_ok(ds4_gpu_tensor_write(g_logits, 0, logits,
                                        N_EXPERT * sizeof(float)),
                   "tie logits write");
        require_ok(ds4_gpu_qwen35_moe_route_tensor(g_sel, g_w, g_logits, N_EXPERT,
                                                   N_USED, 1.0f, 1),
                   "tie route");
        require_ok(ds4_gpu_tensor_read(g_sel, 0, idx, N_USED * sizeof(int32_t)), "tie sel read");
        require_ok(ds4_gpu_tensor_read(g_w, 0, w, N_USED * sizeof(float)), "tie w read");
        for (uint32_t t = 0; t < N_USED; t++) {
            require_ok(idx[t] == t, "uniform tie keeps the lower index");
            require_ok(fabsf(w[t] - 1.0f / (float)N_USED) < 1e-6f,
                       "uniform tie weight is 1/k");
        }
        ds4_gpu_tensor_free(g_w);
        ds4_gpu_tensor_free(g_sel);
        ds4_gpu_tensor_free(g_logits);
    }

    /* 3. A constructed top tie: experts 10 and 200 share the maximum.  The
     *    lower index wins the first slot. */
    {
        static float logits[N_EXPERT];
        for (uint32_t e = 0; e < N_EXPERT; e++)
            logits[e] = synth(7u, e, 0.25f);
        logits[10] = 5.0f;
        logits[200] = 5.0f;
        uint32_t idx[32];
        float w[32];
        ds4_gpu_tensor *g_logits = ds4_gpu_tensor_alloc(N_EXPERT * sizeof(float));
        ds4_gpu_tensor *g_sel = ds4_gpu_tensor_alloc(N_USED * sizeof(int32_t));
        ds4_gpu_tensor *g_w = ds4_gpu_tensor_alloc(N_USED * sizeof(float));
        require_ok(g_logits && g_sel && g_w, "top-tie tensor allocation");
        require_ok(ds4_gpu_tensor_write(g_logits, 0, logits,
                                        N_EXPERT * sizeof(float)),
                   "top-tie logits write");
        require_ok(ds4_gpu_qwen35_moe_route_tensor(g_sel, g_w, g_logits, N_EXPERT,
                                                   N_USED, 1.0f, 1),
                   "top-tie route");
        require_ok(ds4_gpu_tensor_read(g_sel, 0, idx, N_USED * sizeof(int32_t)),
                   "top-tie sel read");
        require_ok(ds4_gpu_tensor_read(g_w, 0, w, N_USED * sizeof(float)), "top-tie w read");
        require_ok(idx[0] == 10 && idx[1] == 200,
                   "top tie resolves to the lower index first");
        ds4_gpu_tensor_free(g_w);
        ds4_gpu_tensor_free(g_sel);
        ds4_gpu_tensor_free(g_logits);
    }

    /* 4. k == n == 8: every expert selected exactly once, ordered by
     *    descending probability, weights renormalised to sum 1. */
    {
        static float logits[N_USED];
        for (uint32_t e = 0; e < N_USED; e++) logits[e] = synth(9u, e, 3.0f);
        check_case("qwen35 router k==n (8 experts, k=8)", logits, 1, N_USED,
                   N_USED, 1.0f, 1e-6f);

        uint32_t idx[32];
        float w[32];
        ds4_gpu_tensor *g_logits = ds4_gpu_tensor_alloc(N_USED * sizeof(float));
        ds4_gpu_tensor *g_sel = ds4_gpu_tensor_alloc(N_USED * sizeof(int32_t));
        ds4_gpu_tensor *g_w = ds4_gpu_tensor_alloc(N_USED * sizeof(float));
        require_ok(g_logits && g_sel && g_w, "k==n tensor allocation");
        require_ok(ds4_gpu_tensor_write(g_logits, 0, logits,
                                        N_USED * sizeof(float)),
                   "k==n logits write");
        require_ok(ds4_gpu_qwen35_moe_route_tensor(g_sel, g_w, g_logits, N_USED,
                                                   N_USED, 1.0f, 1),
                   "k==n route");
        require_ok(ds4_gpu_tensor_read(g_sel, 0, idx, N_USED * sizeof(int32_t)), "k==n sel read");
        require_ok(ds4_gpu_tensor_read(g_w, 0, w, N_USED * sizeof(float)), "k==n w read");
        bool seen[N_USED] = { false };
        float wsum = 0.0f;
        for (uint32_t t = 0; t < N_USED; t++) {
            require_ok(idx[t] >= 0 && idx[t] < (int32_t)N_USED,
                       "k==n index in range");
            require_ok(!seen[idx[t]], "k==n selects each expert exactly once");
            seen[idx[t]] = true;
            wsum += w[t];
        }
        require_ok(fabsf(wsum - 1.0f) < 1e-5f, "k==n weights sum to 1");
        ds4_gpu_tensor_free(g_w);
        ds4_gpu_tensor_free(g_sel);
        ds4_gpu_tensor_free(g_logits);
    }

    /* 5. expert_weight_scale is applied after renormalisation. */
    {
        static float logits[N_EXPERT];
        for (uint32_t e = 0; e < N_EXPERT; e++)
            logits[e] = synth(11u, e, 1.5f);
        check_case("qwen35 router weight scale", logits, 1, N_EXPERT, N_USED,
                   2.5f, 1e-6f);
    }

    ds4_gpu_cleanup();
    puts("qwen35 MoE router Metal: PASS");
    return 0;
}
