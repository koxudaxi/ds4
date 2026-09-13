/* Test-only driver for the O4 Task 2.1 per-layer oracle.
 *
 * Dumps the engine's position-0 hidden state at each llama.cpp node boundary
 * for a caller-supplied token sequence, in the format the comparison script
 * expects.  Not a gate; it exists so the per-layer error curve can be computed
 * without duplicating any layer math.
 *
 *   ./tests/test_qwen35_dump OUT.txt 16 17 18 19 20 21 22
 *
 * Model: DS4_ORNITH_MODEL (default $HOME/models/Ornith-1.5-35B-Q4_K_M.gguf).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int ds4_test_qwen35_dump_layers(const char *gguf, const uint32_t *tokens,
                                uint32_t n_tokens, const char *path);

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s OUT.txt TOKEN [TOKEN ...]\n", argv[0]);
        return 2;
    }
    const char *out = argv[1];

    const char *model = getenv("DS4_ORNITH_MODEL");
    char model_default[1024];
    if (!model || !model[0]) {
        const char *home = getenv("HOME");
        snprintf(model_default, sizeof(model_default),
                 "%s/models/Ornith-1.5-35B-Q4_K_M.gguf", home ? home : ".");
        model = model_default;
    }

    uint32_t toks[8192];
    int n = 0;
    for (int i = 2; i < argc && n < (int)(sizeof(toks) / sizeof(toks[0])); i++) {
        toks[n++] = (uint32_t)strtoul(argv[i], NULL, 10);
    }

    const int rc = ds4_test_qwen35_dump_layers(model, toks, (uint32_t)n, out);
    if (rc != 0) fprintf(stderr, "dump failed: rc=%d\n", rc);
    return rc;
}
