/* Engine-tier acceptance test for the O6 persistent qwen35moe CPU forward.
 *
 * Loads the real Ornith-1.5-35B Q4_K_M GGUF once through the test entry
 * `ds4_test_qwen35_session_run`, runs two tokens through the production
 * persistent forward, and asserts the logits after the second token match the
 * O4 test forward (`ds4_test_qwen35_forward_logits_ref`) for the same two
 * tokens.  The equality is the point: the session consumes token 0, keeps its
 * GQA K/V and GDN state, then consumes token 1 at position 1.
 *
 * It then plants the persistence failure: running the same two tokens with the
 * state reset between them must break the equality, which shows the match is
 * produced by the carried state and not by a re-prefill.
 *
 * The two tokens are the first case of the frozen O0 fixture
 * (tests/test-vectors/ornith-1.5-35b/manifest.json, case cjk_chinese).
 *
 * Environment:
 *   DS4_ORNITH_MODEL   default $HOME/models/Ornith-1.5-35B-Q4_K_M.gguf
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-session
 *   ./external/ds4/tests/test_qwen35_session
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int ds4_test_qwen35_session_run(const char *gguf, const uint32_t *tokens,
                                uint32_t n_tokens, bool reset_between,
                                float *logits_out);
int ds4_test_qwen35_forward_logits_ref(const char *gguf, const uint32_t *tokens,
                                       uint32_t n_tokens,
                                       const uint32_t *continuation,
                                       uint32_t n_steps,
                                       float *logits_out,
                                       uint32_t *greedy_out);

/* DS4_SHAPE_ORNITH15 n_vocab / geometry. */
#define ORNITH_N_VOCAB 248320u

/* Fixed equality bound, chosen before the first run: the two paths perform the
 * same arithmetic in the same order on the same dequantised bytes, so the
 * expectation is exact equality; the bound only absorbs a compiler that
 * reassociates one call site differently. */
#define SESSION_EQUAL_BOUND 1.0e-5f

/* The plant must shift the logits by much more than the bound.  Resetting the
 * state removes both the KV prefix and the GDN recurrence; the resulting
 * logit shift is O(1) or larger on this model. */
#define SESSION_DIVERGE_MIN 1.0e-2f

static uint32_t argmax(const float *v, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
    float worst = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

static double now_secs(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

int main(void) {
    const char *model = getenv("DS4_ORNITH_MODEL");
    char model_default[1024];
    if (!model || !model[0]) {
        const char *home = getenv("HOME");
        snprintf(model_default, sizeof(model_default),
                 "%s/models/Ornith-1.5-35B-Q4_K_M.gguf", home ? home : ".");
        model = model_default;
    }
    if (access(model, R_OK) != 0) {
        fprintf(stderr, "SKIP: model not readable: %s\n", model);
        return 0;
    }

    /* First case of the frozen O0 fixture. */
    const uint32_t tokens[2] = { 99986u, 99449u };
    const uint32_t continuation[1] = { tokens[1] };

    float *ref = calloc(ORNITH_N_VOCAB, sizeof(float));
    float *sess = calloc(ORNITH_N_VOCAB, sizeof(float));
    float *plant = calloc(ORNITH_N_VOCAB, sizeof(float));
    uint32_t ref_greedy = 0;
    if (!ref || !sess || !plant) return 2;

    printf("model: %s\n", model);
    printf("tokens: %u %u\n", tokens[0], tokens[1]);

    double t0 = now_secs();
    const int rrc = ds4_test_qwen35_forward_logits_ref(
            model, tokens, 2u, continuation, 1u, ref, &ref_greedy);
    double t1 = now_secs();
    if (rrc != 0) {
        fprintf(stderr, "FAIL: reference forward rc=%d\n", rrc);
        return 1;
    }

    const int src = ds4_test_qwen35_session_run(model, tokens, 2u, false, sess);
    double t2 = now_secs();
    if (src != 0) {
        fprintf(stderr, "FAIL: session forward rc=%d\n", src);
        return 1;
    }

    const int prc = ds4_test_qwen35_session_run(model, tokens, 2u, true, plant);
    double t3 = now_secs();
    if (prc != 0) {
        fprintf(stderr, "FAIL: reset session forward rc=%d\n", prc);
        return 1;
    }

    const float ref_vs_sess = max_abs_diff(ref, sess, ORNITH_N_VOCAB);
    const float ref_vs_plant = max_abs_diff(ref, plant, ORNITH_N_VOCAB);
    const uint32_t ref_arg = argmax(ref, ORNITH_N_VOCAB);
    const uint32_t sess_arg = argmax(sess, ORNITH_N_VOCAB);

    printf("reference forward: %.1fs, session: %.1fs, reset plant: %.1fs\n",
           t1 - t0, t2 - t1, t3 - t2);
    printf("second-token argmax: reference %u, session %u\n", ref_arg, sess_arg);
    printf("|session - reference|_inf = %.9g (bound %.1e)\n",
           ref_vs_sess, SESSION_EQUAL_BOUND);
    printf("|reset   - reference|_inf = %.9g (min  %.1e)\n",
           ref_vs_plant, SESSION_DIVERGE_MIN);

    int failed = 0;
    if (ref_vs_sess > SESSION_EQUAL_BOUND) {
        printf("FAIL: persistent session does not match the O4 forward\n");
        failed = 1;
    }
    if (ref_arg != sess_arg) {
        printf("FAIL: second-token argmax differs\n");
        failed = 1;
    }
    if (ref_vs_plant <= SESSION_DIVERGE_MIN) {
        printf("FAIL: reset state still matches -- state is not load-bearing\n");
        failed = 1;
    }
    if (failed) {
        free(ref); free(sess); free(plant);
        return 1;
    }
    printf("PASS: persistent session matches the O4 forward; reset breaks it\n");
    free(ref);
    free(sess);
    free(plant);
    return 0;
}
