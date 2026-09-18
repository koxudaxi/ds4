/* Opt-in regression: streamed Laguna must evaluate chat-prefix tokens before
 * any batch prefill has installed a full-model map. Run with a Laguna GGUF. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens tokens = {0};
    float *logits = NULL;
    ds4_engine_options options = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .n_threads = 1, .context_size = 512, .prefill_chunk = 512,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 192,
    };
    int rc = 1;
    char error[512] = {0};
    if (ds4_engine_open(&engine, &options) != 0) goto done;
    const int vocab = ds4_engine_vocab_size(engine);
    if (vocab <= 0 || !(logits = malloc((size_t)vocab * sizeof(float)))) goto done;
    ds4_encode_chat_prompt(engine, NULL, "Hi", DS4_THINK_NONE, &tokens);
    if (tokens.len < 4 || ds4_session_create(&session, engine, 512) != 0) goto done;
    ds4_tokens first = {.v = tokens.v, .len = 1, .cap = 1};
    if (ds4_session_sync(session, &first, error, sizeof(error)) != 0) {
        fprintf(stderr, "FAIL: initial one-token sync: %s\n", error);
        goto done;
    }
    for (int i = 1; i < 4; i++) {
        if (ds4_session_eval(session, tokens.v[i], error, sizeof(error)) != 0) {
            fprintf(stderr, "FAIL: streamed chat-prefix token %d: %s\n", i, error);
            goto done;
        }
        if (ds4_session_argmax(session) < 0) goto done;
        if (ds4_session_copy_logits(session, logits, vocab) != vocab) goto done;
        for (int j = 0; j < vocab; j++) {
            if (!isfinite(logits[j])) {
                fprintf(stderr, "FAIL: nonfinite logit at token %d, id %d\n", i, j);
                goto done;
            }
        }
    }
    puts("PASS: four streamed chat-prefix tokens without batch-prefill views");
    rc = 0;
done:
    free(logits);
    ds4_tokens_free(&tokens);
    if (session) ds4_session_free(session);
    if (engine) ds4_engine_close(engine);
    return rc;
}
