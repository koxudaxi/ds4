/* Engine-tier acceptance test for the qwen35moe CPU forward (O4/O5).
 *
 * Loads the real Ornith-1.5-35B Q4_K_M GGUF through the test hook
 * `ds4_test_qwen35_forward_logits`, runs one O0 fixture case, and asserts the
 * step-0 greedy token equals the frozen llama.cpp reference.  This is the
 * primary gate: greedy-token agreement is exact, not a tolerance.
 *
 * The hook is not declared in ds4.h; its contract is duplicated here.
 *
 * Environment:
 *   DS4_ORNITH_MODEL    default $HOME/models/Ornith-1.5-35B-Q4_K_M.gguf
 *   DS4_ORNITH_FIXTURE  default tests/test-vectors/ornith-1.5-35b
 *   argv[1]             case id (default cjk_chinese)
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-e2e
 *   DS4_TEST_MODEL=... ./external/ds4/tests/test_qwen35_e2e cjk_chinese
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_test_qwen35_forward_logits(const char *gguf, const uint32_t *tokens,
                                   uint32_t n_tokens, uint32_t n_steps,
                                   float *logits_out, uint32_t *greedy_out);

#define VOCAB_MAX 300000u
/* DS4_SHAPE_ORNITH15 n_vocab.  The hook writes exactly this many logits. */
#define ORNITH_N_VOCAB 248320u

/* Fast-math-safe finite test.  This unit is built with -ffast-math
 * (-ffinite-math-only), which lets the compiler fold isfinite() and by-value
 * bit inspection to true, so read the IEEE-754 binary32 exponent bits through
 * a pointer into the logits buffer. */
static int is_finite_logit(const float *p) {
    uint32_t bits;
    memcpy(&bits, (const void *)p, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

static char *read_file(const char *path, long *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1u);
    if (!buf) { fclose(fp); return NULL; }
    if (fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf); fclose(fp); return NULL;
    }
    buf[len] = '\0';
    fclose(fp);
    if (len_out) *len_out = len;
    return buf;
}

/* Extract the token_ids array for `case_id` from manifest.json. */
static uint32_t parse_manifest(const char *text, const char *case_id,
                               uint32_t *out, uint32_t cap) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"id\": \"%s\"", case_id);
    const char *p = strstr(text, needle);
    if (!p) {
        fprintf(stderr, "manifest: case id %s not found\n", case_id);
        return 0;
    }
    p = strstr(p, "\"token_ids\"");
    if (!p) {
        fprintf(stderr, "manifest: token_ids not found for %s\n", case_id);
        return 0;
    }
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    uint32_t n = 0;
    while (*p && *p != ']') {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == ',') p++;
        if (*p == ']') break;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (n >= cap) { fprintf(stderr, "manifest: too many token_ids\n"); return 0; }
        out[n++] = (uint32_t)v;
        p = end;
    }
    return n;
}

/* Find the step-0 selected token for `case_id` in reference.vec. */
static int parse_reference(const char *text, const char *case_id,
                           uint32_t *token_out) {
    char label[256];
    snprintf(label, sizeof(label), "case %s ", case_id);
    const char *p = text;
    while ((p = strstr(p, label)) != NULL) {
        const char *nl = strchr(p, '\n');
        if (!nl) break;
        const char *q = nl + 1;
        if (strncmp(q, "step 0 ", 7) == 0) {
            *token_out = (uint32_t)strtoul(q + 7, NULL, 10);
            return 1;
        }
        p = nl;
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *case_id = argc > 1 ? argv[1] : "cjk_chinese";

    const char *model = getenv("DS4_ORNITH_MODEL");
    char model_default[1024];
    if (!model || !model[0]) {
        const char *home = getenv("HOME");
        snprintf(model_default, sizeof(model_default),
                 "%s/models/Ornith-1.5-35B-Q4_K_M.gguf", home ? home : ".");
        model = model_default;
    }

    const char *fixture = getenv("DS4_ORNITH_FIXTURE");
    if (!fixture || !fixture[0]) fixture = "tests/test-vectors/ornith-1.5-35b";

    char manifest_path[1200];
    char reference_path[1200];
    snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", fixture);
    snprintf(reference_path, sizeof(reference_path), "%s/reference.vec", fixture);

    char *manifest = read_file(manifest_path, NULL);
    char *reference = read_file(reference_path, NULL);
    if (!manifest || !reference) return 2;

    uint32_t tokens[8192];
    uint32_t n_tokens = parse_manifest(manifest, case_id, tokens, 8192);
    if (n_tokens == 0) { fprintf(stderr, "no tokens for %s\n", case_id); return 2; }

    uint32_t expected = 0;
    if (!parse_reference(reference, case_id, &expected)) {
        fprintf(stderr, "reference: case %s not found\n", case_id);
        return 2;
    }

    free(manifest);
    free(reference);

    fprintf(stderr, "case %s: %u prompt tokens, expected step-0 token %u\n",
            case_id, n_tokens, expected);

    float *logits = malloc((size_t)VOCAB_MAX * sizeof(float));
    uint32_t greedy = 0;
    if (!logits) return 2;

    const int rc = ds4_test_qwen35_forward_logits(
            model, tokens, n_tokens, 1, logits, &greedy);
    if (rc != 0) {
        fprintf(stderr, "forward failed: rc=%d\n", rc);
        free(logits);
        return 1;
    }

    /* Finite gate before any value comparison (spec verification bullet). */
    for (uint32_t v = 0; v < ORNITH_N_VOCAB; v++) {
        if (!is_finite_logit(&logits[v])) {
            fprintf(stderr, "FAIL: %s logit %u is not finite (%g)\n",
                    case_id, v, logits[v]);
            free(logits);
            return 1;
        }
    }

    printf("step-0 greedy token: %u (reference %u)\n", greedy, expected);
    free(logits);

    if (greedy != expected) {
        printf("FAIL: %s step-0 greedy mismatch\n", case_id);
        return 1;
    }
    printf("PASS: %s step-0 greedy token matches\n", case_id);
    return 0;
}
