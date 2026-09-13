/* Engine-tier acceptance test for the qwen35moe CPU forward (O4/O5).
 *
 * Loads the real Ornith-1.5-35B Q4_K_M GGUF through the test hook
 * `ds4_test_qwen35_forward_logits`, iterates every case in the frozen O0
 * fixture (`manifest.json` + `reference.vec`), and for each case:
 *   1. asserts every returned logit is finite (before any value comparison),
 *   2. compares each step's greedy token to the reference (the primary gate,
 *      exact, not a tolerance),
 *   3. compares the reference top-k raw logits to ours within a fixed bound.
 *
 * Tolerance is fixed in this file before the first comparison run; see
 * ORNITH_LOGIT_BOUND.  It is not tuned to pass.
 *
 * The hook is not declared in ds4.h; its contract is duplicated here.
 *
 * Environment:
 *   DS4_ORNITH_MODEL    default $HOME/models/Ornith-1.5-35B-Q4_K_M.gguf
 *   DS4_ORNITH_FIXTURE  default tests/test-vectors/ornith-1.5-35b
 *   argv[1]             case id, or "all" (default) to run every case
 *
 * Build/run:
 *   make -C external/ds4 test-qwen35-e2e
 *   DS4_ORNITH_MODEL=... ./external/ds4/tests/test_qwen35_e2e [case|all]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int ds4_test_qwen35_forward_logits(const char *gguf, const uint32_t *tokens,
                                   uint32_t n_tokens, uint32_t n_steps,
                                   float *logits_out, uint32_t *greedy_out);
int ds4_test_qwen35_forward_logits_ref(const char *gguf, const uint32_t *tokens,
                                       uint32_t n_tokens,
                                       const uint32_t *continuation,
                                       uint32_t n_steps,
                                       float *logits_out, uint32_t *greedy_out);

/* DS4_SHAPE_ORNITH15 n_vocab.  The hook writes exactly this many logits/step. */
#define ORNITH_N_VOCAB 248320u

#define MAX_CASES   32
#define MAX_STEPS   16
#define MAX_TOPK    64
#define MAX_TOKENS  8192

/* Fixed top-k logit bound (spec: the tolerance is fixed before the comparison
 * and reported as a worst obs/bound ratio).  Both the engine and llama.cpp read
 * the same Q4_K_M bytes, so there is no weight error.  The engine dequantises to
 * f32 and accumulates in f64 (`matvec_f32_worker`, ds4.c:8973); llama.cpp CPU
 * quantises activations to q8_K (step amax/127, ~0.7% RMS per element) and
 * accumulates in f32.  Per matvec that is ~1% of the output; a random walk over
 * the ~80 matvecs of the 40-layer trunk is ~6-8% of the hidden scale, and the
 * reference top-k are the *extremes* of its distribution, so comparing our
 * values at those indices adds a one-sided selection bias.  On a logit scale of
 * +-16 a 2.0 bound absorbs that (12.5% of the top logit) yet is 4x tighter than
 * the engine's existing local-golden drift bound of 8.0 (tests/ds4_test.c:5655)
 * and far below the O(1)+/O(10) shift a wiring error (rope / expert / residual)
 * produces.  Greedy-token equality remains the decisive gate. */
#define ORNITH_LOGIT_BOUND 2.0f

/* `long_near_context` is 3,800 tokens and the forward is a full O(L^2) causal
 * prefill per step, so it is not run in the default sweep; the driver prints an
 * explicit SKIP with the reason (it is not silently dropped).  The next longest
 * prompt is 35 tokens. */
#define ORNITH_MAX_PROMPT_TOKENS 512u

typedef struct {
    uint32_t id;
    float    logit;
} ref_top;

typedef struct {
    uint32_t selected;
    int      ntop;
    ref_top  top[MAX_TOPK];
} ref_step;

typedef struct {
    char     id[64];
    uint32_t n_tokens;
    uint32_t tokens[MAX_TOKENS];
} manifest_case;

typedef struct {
    char     id[64];
    int      n_steps;
    ref_step steps[MAX_STEPS];
} ref_case;

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

/* Extract every prompt's id + token_ids from manifest.json, in order. */
static int parse_manifest_all(const char *text, manifest_case *cases, int cap) {
    int n = 0;
    const char *p = text;
    while (n < cap) {
        const char *idk = strstr(p, "\"id\": \"");
        if (!idk) break;
        const char *idstart = idk + 7;
        const char *idend = strchr(idstart, '"');
        if (!idend) break;
        const char *nextid = strstr(idend + 1, "\"id\": \"");
        const char *tk = strstr(idend + 1, "\"token_ids\"");
        if (!tk || (nextid && tk > nextid)) { p = idend + 1; continue; }

        manifest_case *c = &cases[n];
        size_t idlen = (size_t)(idend - idstart);
        if (idlen >= sizeof(c->id)) idlen = sizeof(c->id) - 1u;
        memcpy(c->id, idstart, idlen);
        c->id[idlen] = '\0';

        const char *q = strchr(tk, '[');
        if (!q) { p = idend + 1; continue; }
        q++;
        c->n_tokens = 0;
        while (*q && *q != ']') {
            while (*q == ' ' || *q == '\n' || *q == '\t' || *q == ',') q++;
            if (*q == ']') break;
            char *end = NULL;
            unsigned long v = strtoul(q, &end, 10);
            if (end == q) break;
            if (c->n_tokens >= MAX_TOKENS) {
                fprintf(stderr, "manifest: too many token_ids for %s\n", c->id);
                return n;
            }
            c->tokens[c->n_tokens++] = (uint32_t)v;
            q = end;
        }
        if (c->n_tokens == 0) { p = idend + 1; continue; }
        n++;
        p = q;
    }
    return n;
}

/* Parse reference.vec into per-case step selections and top-k logits. */
static int parse_reference_all(const char *text, ref_case *cases, int cap) {
    char *copy = malloc(strlen(text) + 1u);
    if (!copy) return 0;
    strcpy(copy, text);

    int n = 0;
    ref_case *cur = NULL;
    int cur_step = -1;
    char *save = NULL;
    for (char *line = strtok_r(copy, "\n", &save); line;
         line = strtok_r(NULL, "\n", &save)) {
        if (line[0] == '#' || line[0] == '\0') continue;
        if (strncmp(line, "case ", 5) == 0) {
            int steps = 0;
            char id[64] = {0};
            if (sscanf(line + 5, "%63s %d", id, &steps) != 2) continue;
            if (n >= cap) break;
            cur = &cases[n++];
            memset(cur, 0, sizeof(*cur));
            snprintf(cur->id, sizeof(cur->id), "%s", id);
            cur->n_steps = steps > MAX_STEPS ? MAX_STEPS : steps;
            cur_step = -1;
        } else if (!cur) {
            continue;
        } else if (strncmp(line, "step ", 5) == 0) {
            int idx = -1;
            unsigned tok = 0;
            if (sscanf(line + 5, "%d %u", &idx, &tok) != 2) continue;
            if (idx < 0 || idx >= MAX_STEPS) continue;
            cur->steps[idx].selected = (uint32_t)tok;
            if (idx + 1 > cur->n_steps) cur->n_steps = idx + 1;
            cur_step = idx;
        } else if (strncmp(line, "top ", 4) == 0) {
            int rank = -1;
            unsigned tid = 0;
            float logit = 0.0f;
            if (cur_step < 0) continue;
            if (sscanf(line + 4, "%d %u %f", &rank, &tid, &logit) != 3) continue;
            ref_step *st = &cur->steps[cur_step];
            if (st->ntop >= MAX_TOPK) continue;
            st->top[st->ntop].id = (uint32_t)tid;
            st->top[st->ntop].logit = logit;
            st->ntop++;
        }
    }
    free(copy);
    return n;
}

int main(int argc, char **argv) {
    const char *only = argc > 1 && strcmp(argv[1], "all") != 0 ? argv[1] : NULL;

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

    manifest_case *mc = calloc(MAX_CASES, sizeof(*mc));
    ref_case *rc = calloc(MAX_CASES, sizeof(*rc));
    if (!mc || !rc) return 2;
    const int n_manifest = parse_manifest_all(manifest, mc, MAX_CASES);
    const int n_ref = parse_reference_all(reference, rc, MAX_CASES);
    free(manifest);
    free(reference);
    if (n_manifest == 0 || n_ref == 0) {
        fprintf(stderr, "fixture: parsed %d manifest / %d reference cases\n",
                n_manifest, n_ref);
        return 2;
    }

    printf("mode: %s\n",
           getenv("DS4_ORNITH_AUTO") ? "auto-regressive (generation order)"
                                     : "reference-prefix (forced, per-step)");
    printf("case                     steps  greedy   worst top-k   secs  status\n");
    printf("-------------------------------------------- -------------------\n");

    int failed = 0, skipped = 0, ran = 0, n_incomplete = 0;
    int total_steps = 0, total_greedy = 0;
    float worst_all = 0.0f;
    char worst_case[64] = "";

    for (int i = 0; i < n_manifest; i++) {
        const manifest_case *m = &mc[i];
        if (only && strcmp(m->id, only) != 0) continue;

        const ref_case *r = NULL;
        for (int j = 0; j < n_ref; j++) {
            if (strcmp(rc[j].id, m->id) == 0) { r = &rc[j]; break; }
        }
        if (!r) {
            printf("%-24s -- reference case not found\n", m->id);
            failed++;
            continue;
        }

        if (m->n_tokens > ORNITH_MAX_PROMPT_TOKENS &&
            !getenv("DS4_ORNITH_ALLOW_LONG")) {
            printf("%-24s SKIP  %u prompt tokens > %u (O(L^2) prefill per step)\n",
                   m->id, m->n_tokens, ORNITH_MAX_PROMPT_TOKENS);
            skipped++;
            continue;
        }

        int n_steps = r->n_steps;
        {
            const char *ms = getenv("DS4_ORNITH_MAX_STEPS");
            if (ms) {
                int cap = atoi(ms);
                if (cap >= 1 && cap < n_steps) n_steps = cap;
            }
        }
        float *logits = calloc((size_t)n_steps * ORNITH_N_VOCAB, sizeof(float));
        uint32_t *greedy = calloc((size_t)n_steps, sizeof(uint32_t));
        uint32_t *ref_cont = calloc((size_t)n_steps, sizeof(uint32_t));
        if (!logits || !greedy || !ref_cont) {
            free(logits); free(greedy); free(ref_cont); return 2;
        }
        for (int s = 0; s < n_steps; s++) ref_cont[s] = r->steps[s].selected;

        /* Default: score each step against the prefix the reference used (its
         * own greedy continuation), so a near-tie at step k does not corrupt
         * step k+1.  DS4_ORNITH_AUTO=1 instead lets the engine feed its own
         * greedy token, which is the generation order and has the same first
         * divergence. */
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        const int hrc = getenv("DS4_ORNITH_AUTO")
                ? ds4_test_qwen35_forward_logits(
                          model, m->tokens, m->n_tokens, (uint32_t)n_steps,
                          logits, greedy)
                : ds4_test_qwen35_forward_logits_ref(
                          model, m->tokens, m->n_tokens, ref_cont,
                          (uint32_t)n_steps, logits, greedy);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double secs = (double)(t1.tv_sec - t0.tv_sec) +
                            (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        if (hrc != 0) {
            printf("%-24s FAIL  forward rc=%d\n", m->id, hrc);
            failed++;
            free(logits);
            free(greedy);
            free(ref_cont);
            continue;
        }
        ran++;

        /* Finite gate first (spec verification bullet). */
        int finite = 1;
        for (size_t v = 0; v < (size_t)n_steps * ORNITH_N_VOCAB; v++) {
            if (!is_finite_logit(&logits[v])) {
                fprintf(stderr, "%s: non-finite logit at flat index %zu\n",
                        m->id, v);
                finite = 0;
                break;
            }
        }
        if (!finite) {
            printf("%-24s FAIL  non-finite logits\n", m->id);
            failed++;
            free(logits);
            free(greedy);
            free(ref_cont);
            continue;
        }

        int agree = 0, first_div = -1;
        float worst_ratio = 0.0f;
        for (int s = 0; s < n_steps; s++) {
            total_steps++;
            if (greedy[s] == r->steps[s].selected) {
                agree++;
                total_greedy++;
            } else if (first_div < 0) {
                first_div = s;
            }
            const float *row = logits + (size_t)s * ORNITH_N_VOCAB;
            float step_worst = 0.0f;
            for (int k = 0; k < r->steps[s].ntop; k++) {
                const uint32_t tid = r->steps[s].top[k].id;
                if (tid >= ORNITH_N_VOCAB) continue;
                const float diff = fabsf(row[tid] - r->steps[s].top[k].logit);
                const float ratio = diff / ORNITH_LOGIT_BOUND;
                if (ratio > worst_ratio) worst_ratio = ratio;
                if (diff > step_worst) step_worst = diff;
            }
            if (getenv("DS4_ORNITH_VERBOSE")) {
                const int ntop = r->steps[s].ntop;
                fprintf(stderr, "  step %d ours=%u ref=%u step worst |d|=%.4f\n",
                        s, greedy[s], r->steps[s].selected, step_worst);
                for (int k = 0; k < ntop; k++) {
                    const uint32_t tid = r->steps[s].top[k].id;
                    if (tid >= ORNITH_N_VOCAB) continue;
                    fprintf(stderr, "    #%2d tok %6u ref %9.4f ours %9.4f d %8.4f\n",
                            k, tid, r->steps[s].top[k].logit, row[tid],
                            row[tid] - r->steps[s].top[k].logit);
                }
            }
        }
        if (worst_ratio > worst_all) {
            worst_all = worst_ratio;
            snprintf(worst_case, sizeof(worst_case), "%s", m->id);
        }
        if (agree < n_steps) n_incomplete++;

        const int ok = (agree == n_steps) && (worst_ratio <= 1.0f);
        if (!ok) failed++;

        printf("%-24s %5d  %2d/%2d    %10.4f   %6.1f  %s%s\n",
               m->id, n_steps, agree, n_steps, worst_ratio, secs,
               ok ? "PASS" : "FAIL",
               first_div >= 0 ? " (greedy diverged)" : "");

        free(logits);
        free(greedy);
        free(ref_cont);
    }

    printf("-------------------------------------------- -------------------\n");
    printf("ran %d cases, %d skipped", ran, skipped);
    if (ran) printf(", greedy %d/%d steps", total_greedy, total_steps);
    printf("\n");
    if (ran)
        printf("worst top-k ratio: %.4f (%s), bound %.2f\n",
               worst_all, worst_case[0] ? worst_case : "-", ORNITH_LOGIT_BOUND);
    if (n_incomplete)
        printf("cases with a greedy divergence: %d\n", n_incomplete);

    free(mc);
    free(rc);
    if (only && ran == 0) {
        fprintf(stderr, "case %s not found in the manifest\n", only);
        return 2;
    }
    if (failed) {
        printf("FAIL: %d case(s) off\n", failed);
        return 1;
    }
    printf("PASS: every run case matches the O0 reference\n");
    return 0;
}
