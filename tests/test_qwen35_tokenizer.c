/* Model-free tests for the qwen35 (Ornith 1.5 35B / Qwen3.5-MoE) loader and
 * pre-tokenizer.
 *
 * These exist because the GLM 5.3 review's Critical finding was a check that
 * never ran: the layer-typing rule and the shape-validation walk were each
 * exercised only by one run against one 20 GiB artifact, and the model is too
 * large and too absent to put in the default tier.  Everything here is pure
 * logic: no model, no GPU, no processes, no I/O.
 *
 * The pre-tokenizer is additionally verified end-to-end against llama.cpp's
 * `llama-tokenize` by an external harness (153/153 prompts); the piece
 * expectations below are the subset that can be stated from the reference
 * semantics alone, so they hold even where llama.cpp is not installed.
 *
 * Build/run: make tests/test_qwen35_tokenizer && ./tests/test_qwen35_tokenizer
 */
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ds4.h"

/* Hooks exported by ds4.c under -DDS4_TEST_HOOKS. */
bool     ds4_test_qwen35_layer_is_linear(uint32_t il);
uint32_t ds4_test_qwen35_ucat(uint32_t cp);
uint64_t ds4_test_qwen35_pieces(const char *text, uint64_t *bounds, uint64_t cap);
uint32_t ds4_test_qwen35_validate_end(uint32_t layer_end, bool mtp_bound);
bool     ds4_test_qwen35_dense_quant(uint32_t type);
void     ds4_test_qwen35_gap_reset(void);
void     ds4_test_qwen35_gap_add(uint32_t type);
uint32_t ds4_test_qwen35_gap_total(void);
void     ds4_test_qwen35_render_chat(const char *system, const char *prompt,
                                     bool thinking, char *out, size_t cap);
void     ds4_test_qwen35_render_turn(const char *role, const char *content,
                                     char *out, size_t cap);
uint32_t ds4_test_qwen35_moe_route(const float *logits, uint32_t n_expert, uint32_t top_k,
                                   uint32_t *indices, float *weights);
float    ds4_test_qwen35_shared_expert_gate(float dot);

/* GGML tensor type ids the quant-gap decision turns on. */
#define T_F32   0u
#define T_Q4_0  2u
#define T_Q8_0  8u
#define T_Q4_K 12u
#define T_Q6_K 14u

/* Unicode general categories, as produced by the generated table. */
#define CAT_OTHER  0u
#define CAT_LETTER 1u
#define CAT_MARK   2u
#define CAT_NUMBER 3u

#define QWEN35_N_LAYER 41u

static int failures;

static void check(bool ok, const char *what) {
    if (ok) return;
    printf("FAIL  %s\n", what);
    failures++;
}

static void check_u32(uint32_t got, uint32_t want, const char *what) {
    if (got == want) return;
    printf("FAIL  %s: got %u, want %u\n", what, got, want);
    failures++;
}

static void test_layer_typing(void) {
    uint32_t linear = 0;
    uint32_t full = 0;

    for (uint32_t il = 0; il < QWEN35_N_LAYER; il++) {
        const bool is_linear = ds4_test_qwen35_layer_is_linear(il);
        /* Trunk: full attention every 4th block (indices 3, 7, ... 39).  The
         * MTP block (40) is a full-attention block. */
        const bool expect_full = (il % 4u == 3u) || (il + 1u == QWEN35_N_LAYER);
        if (is_linear == expect_full) {
            printf("FAIL  layer %u: linear=%d, expected %s\n",
                   il, (int)is_linear, expect_full ? "full" : "linear");
            failures++;
        }
        if (is_linear) linear++; else full++;
    }

    check_u32(linear, 30u, "30 gated-delta-net layers");
    check_u32(full, 11u, "11 full-attention layers (10 trunk + MTP)");

    /* Out of range must not claim to be a linear layer. */
    check(!ds4_test_qwen35_layer_is_linear(QWEN35_N_LAYER),
          "layer 41 is not a linear layer");
}

static void test_unicode_categories(void) {
    check_u32(ds4_test_qwen35_ucat('a'), CAT_LETTER, "ASCII lowercase letter");
    check_u32(ds4_test_qwen35_ucat('Z'), CAT_LETTER, "ASCII uppercase letter");
    check_u32(ds4_test_qwen35_ucat('0'), CAT_NUMBER, "ASCII digit");
    check_u32(ds4_test_qwen35_ucat(' '), CAT_OTHER, "ASCII space is not L/M/N");
    check_u32(ds4_test_qwen35_ucat('!'), CAT_OTHER, "ASCII punctuation");

    check_u32(ds4_test_qwen35_ucat(0x4e2du), CAT_LETTER, "CJK ideograph is Lo");
    check_u32(ds4_test_qwen35_ucat(0x3042u), CAT_LETTER, "hiragana is Lo");
    check_u32(ds4_test_qwen35_ucat(0x20000u), CAT_LETTER, "CJK ext B is Lo");
    check_u32(ds4_test_qwen35_ucat(0x0301u), CAT_MARK, "combining acute is Mn");
    check_u32(ds4_test_qwen35_ucat(0x093fu), CAT_MARK, "Devanagari sign i is Mc");
    check_u32(ds4_test_qwen35_ucat(0x0661u), CAT_NUMBER, "Arabic-Indic digit is Nd");
    check_u32(ds4_test_qwen35_ucat(0x00b2u), CAT_NUMBER, "superscript two is No");
    check_u32(ds4_test_qwen35_ucat(0x2167u), CAT_NUMBER, "Roman numeral is Nl");
    check_u32(ds4_test_qwen35_ucat(0xff21u), CAT_LETTER, "fullwidth A is Lu");
    check_u32(ds4_test_qwen35_ucat(0x1f600u), CAT_OTHER, "emoji is So, not L/M/N");
    check_u32(ds4_test_qwen35_ucat(0x2014u), CAT_OTHER, "em dash is Pd");

    /* Table bounds: the first and last generated entries must be reachable. */
    check_u32(ds4_test_qwen35_ucat(0x000aau), CAT_LETTER, "first table entry");
    check_u32(ds4_test_qwen35_ucat(0x10ffffu), CAT_OTHER, "above the table");
}

/* Split `text` and compare against an expected piece count and a joined form
 * with '|' separators, which makes a mismatch readable. */
static void check_pieces(const char *text, const char *expected) {
    uint64_t bounds[64];
    const uint64_t pieces = ds4_test_qwen35_pieces(text, bounds, 64);

    if (pieces == UINT64_MAX) {
        printf("FAIL  pieces(%s): buffer too small\n", text);
        failures++;
        return;
    }

    char got[256];
    size_t n = 0;
    for (uint64_t i = 0; i < pieces && n + 2 < sizeof(got); i++) {
        const uint64_t len = bounds[i + 1u] - bounds[i];
        if (i) got[n++] = '|';
        for (uint64_t j = 0; j < len && n + 2 < sizeof(got); j++) {
            got[n++] = text[bounds[i] + j];
        }
    }
    got[n] = '\0';

    if (pieces > 0 && bounds[0] != 0) {
        printf("FAIL  pieces(%s): first boundary is %llu, not 0\n",
               text, (unsigned long long)bounds[0]);
        failures++;
    }
    if (pieces > 0 && bounds[pieces] != strlen(text)) {
        printf("FAIL  pieces(%s): last boundary is %llu, not %zu\n",
               text, (unsigned long long)bounds[pieces], strlen(text));
        failures++;
    }
    if (strcmp(got, expected) != 0) {
        printf("FAIL  pieces: got \"%s\", want \"%s\"\n", got, expected);
        failures++;
    }
}

static void test_piece_splitting(void) {
    /* Expectations follow the reference pre-tokenizer's alternative order:
     * \p{N} is a single character, an optional non-L/N prefix joins a letter
     * run, a newline run is its own piece, and \s+(?!\S) leaves one space
     * behind. */
    check_pieces("abc", "abc");
    check_pieces("a b", "a| b");
    check_pieces("a  b", "a| | b");
    check_pieces("123", "1|2|3");
    check_pieces("a1", "a|1");
    check_pieces("don't", "don|'t");
    check_pieces("it's", "it|'s");
    check_pieces("we're", "we|'re");
    check_pieces("a\nb", "a|\n|b");
    check_pieces("a\n\nb", "a|\n\n|b");
    check_pieces("a\r\nb", "a|\r\n|b");
    check_pieces("hello!", "hello|!");
    check_pieces("!!!", "!!!");
    check_pieces("a!!b", "a|!!|b");
    check_pieces("\xE4\xB8\xAD\xE6\x96\x87", "\xE4\xB8\xAD\xE6\x96\x87");  /* CJK */
    check_pieces("", "");

    /* Every boundary must be monotonically increasing and within the text. */
    {
        static const char *samples[] = {
            "  leading", "trailing  ", "a\tb", "x  \n  y", "\n", "   ",
            "Mixed: 你好 123 abc", "emoji \xF0\x9F\x91\x8D here",
        };
        for (size_t s = 0; s < sizeof(samples) / sizeof(samples[0]); s++) {
            uint64_t bounds[128];
            const uint64_t pieces = ds4_test_qwen35_pieces(samples[s], bounds, 128);
            const uint64_t len = strlen(samples[s]);
            if (pieces == UINT64_MAX) {
                printf("FAIL  monotonic(%s): buffer too small\n", samples[s]);
                failures++;
                continue;
            }
            for (uint64_t i = 0; i < pieces; i++) {
                if (bounds[i] >= bounds[i + 1u] || bounds[i + 1u] > len) {
                    printf("FAIL  monotonic(%s): boundary %llu..%llu of %llu\n",
                           samples[s], (unsigned long long)bounds[i],
                           (unsigned long long)bounds[i + 1u],
                           (unsigned long long)len);
                    failures++;
                    break;
                }
            }
        }
    }
}

static void test_validation_coverage(void) {
    /* The finding this guards: the walk stopped at the executable range, so the
     * MTP block was bound and never checked. */
    check_u32(ds4_test_qwen35_validate_end(39u, true), 40u,
              "walk reaches the MTP block when bound");
    check_u32(ds4_test_qwen35_validate_end(39u, false), 39u,
              "walk stops at the executable range when the MTP block is not bound");
}

static void test_quant_gap_report(void) {
    /* The whitelist, and the type that is outside it. */
    check(ds4_test_qwen35_dense_quant(T_Q8_0), "q8_0 is a dense quant type");
    check(ds4_test_qwen35_dense_quant(T_Q4_K), "q4_K is a dense quant type");
    check(ds4_test_qwen35_dense_quant(T_Q4_0), "q4_0 is a dense quant type");
    check(!ds4_test_qwen35_dense_quant(T_Q6_K), "q6_k is not a dense quant type");
    check(!ds4_test_qwen35_dense_quant(T_F32), "f32 is not a dense quant type");

    /* And that every dense tensor is counted, which the first version of the
     * report got wrong: it covered a subset of the tensors and named 20 of the
     * 43 in the artifact. */
    ds4_test_qwen35_gap_reset();
    ds4_test_qwen35_gap_add(T_Q8_0);
    check_u32(ds4_test_qwen35_gap_total(), 0u, "whitelisted type is not counted");

    ds4_test_qwen35_gap_add(T_Q6_K);
    ds4_test_qwen35_gap_add(T_Q6_K);
    ds4_test_qwen35_gap_add(T_F32);
    check_u32(ds4_test_qwen35_gap_total(), 3u, "every non-whitelisted type is counted");

    ds4_test_qwen35_gap_reset();
    check_u32(ds4_test_qwen35_gap_total(), 0u, "reset clears the accumulator");
}

static void test_chat_rendering(void) {
    /* Expectations are the GGUF's own template (tokenizer.chat_template),
     * rendered.  The whole chat result is verified against Jinja + llama.cpp
     * by an external harness; these are the invariants that must hold without
     * either of those present. */
    char out[1024];

    ds4_test_qwen35_render_chat("", "hi", true, out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>user\nhi<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n") == 0,
          "no system prompt omits the system turn");

    ds4_test_qwen35_render_chat("", "hi", false, out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>user\nhi<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n") == 0,
          "disabled thinking pre-closes an empty think block");

    ds4_test_qwen35_render_chat("sys", "hi", true, out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>system\nsys<|im_end|>\n"
                 "<|im_start|>user\nhi<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n") == 0,
          "system turn is emitted before the user turn");

    /* The template applies |trim to content, so padding must not survive. */
    ds4_test_qwen35_render_chat("  sys  ", "  hi  ", true, out, sizeof(out));
    check(strstr(out, "system\nsys<|im_end|>") != NULL,
          "system content is trimmed");
    check(strstr(out, "user\nhi<|im_end|>") != NULL,
          "user content is trimmed");

    /* Interior newlines are content and must be preserved. */
    ds4_test_qwen35_render_chat("", "a\nb", true, out, sizeof(out));
    check(strstr(out, "user\na\nb<|im_end|>") != NULL,
          "interior newlines survive trimming");

    /* An all-whitespace prompt renders as an empty user turn, not a space. */
    ds4_test_qwen35_render_chat("", "   ", true, out, sizeof(out));
    check(strstr(out, "user\n<|im_end|>") != NULL,
          "all-whitespace prompt becomes an empty turn");
}

static void test_turn_rendering(void) {
    /* Per-turn rendering for the roles the chat API can express.  The full
     * multi-turn sequence is diffed against the template by msgcmp.py; these
     * are the invariants that must hold without Jinja present. */
    char out[4096];

    ds4_test_qwen35_render_turn("user", "hi", out, sizeof(out));
    check(strcmp(out, "<|im_start|>user\nhi<|im_end|>\n") == 0,
          "user turn");

    ds4_test_qwen35_render_turn("system", "be terse", out, sizeof(out));
    check(strcmp(out, "<|im_start|>system\nbe terse<|im_end|>\n") == 0,
          "system turn");

    /* Assistant with no reasoning: the template opens and closes an empty
     * <think> block before the answer. */
    ds4_test_qwen35_render_turn("assistant", "the answer", out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>assistant\n<think>\n\n</think>\n\nthe answer"
                 "<|im_end|>\n") == 0,
          "assistant turn without reasoning");

    /* Assistant carrying its reasoning: the template splits on the last
     * </think> and drops the newlines adjacent to it. */
    ds4_test_qwen35_render_turn("assistant",
                                "<think>\nwhy\n</think>\nthe answer",
                                out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>assistant\n<think>\nwhy\n</think>\n\nthe answer"
                 "<|im_end|>\n") == 0,
          "assistant turn with reasoning is split on </think>");

    ds4_test_qwen35_render_turn("tool", "42", out, sizeof(out));
    check(strcmp(out,
                 "<|im_start|>user\n<tool_response>\n42\n</tool_response>"
                 "<|im_end|>\n") == 0,
          "tool turn renders as a user turn with a tool_response block");

    ds4_test_qwen35_render_turn("user", "  padded  ", out, sizeof(out));
    check(strcmp(out, "<|im_start|>user\npadded<|im_end|>\n") == 0,
          "turn content is trimmed");
}

static void test_moe_routing(void) {
    /* Reference is the template-independent router in transformers:
     * softmax over all experts, top-k, renormalise.  routecmp.py checks this
     * against a pure-Python reference on random logits; these are the cases
     * that can be stated exactly. */
    float logits[8];
    uint32_t idx[8];
    float w[8];

    /* Uniform logits: every expert equal, so the top-k is the k lowest indices
     * and the renormalised weights are exactly 1/k. */
    for (int i = 0; i < 8; i++) logits[i] = 0.5f;
    uint32_t n = ds4_test_qwen35_moe_route(logits, 8, 3, idx, w);
    check_u32(n, 3u, "route returns k experts");
    check(idx[0] == 0 && idx[1] == 1 && idx[2] == 2,
          "uniform logits select the lowest indices in order");
    for (int i = 0; i < 3; i++) {
        check(fabsf(w[i] - 1.0f / 3.0f) < 1e-6f,
              "uniform logits renormalise to 1/k");
    }

    /* A dominant expert takes most of the probability and comes first. */
    for (int i = 0; i < 8; i++) logits[i] = 0.0f;
    logits[5] = 10.0f;
    logits[2] = 5.0f;
    n = ds4_test_qwen35_moe_route(logits, 8, 2, idx, w);
    check_u32(n, 2u, "route with k=2");
    check(idx[0] == 5 && idx[1] == 2, "experts come back in descending probability");
    {
        float s = w[0] + w[1];
        check(fabsf(s - 1.0f) < 1e-6f, "selected weights are renormalised to sum 1");
        check(w[0] > w[1], "the dominant expert carries the larger weight");
    }

    /* Top-k larger than the expert count is clamped, not an error. */
    n = ds4_test_qwen35_moe_route(logits, 4, 9, idx, w);
    check_u32(n, 4u, "top_k is clamped to the expert count");

    /* Top-k equal to the expert count: every expert is selected exactly once,
     * ordered by descending probability. */
    for (int i = 0; i < 8; i++) logits[i] = 0.0f;
    logits[6] = 3.0f;
    logits[1] = 1.0f;
    n = ds4_test_qwen35_moe_route(logits, 8, 8, idx, w);
    check_u32(n, 8u, "top_k == n selects every expert");
    check(idx[0] == 6 && idx[1] == 1,
          "k == n orders by descending probability");
    {
        bool seen[8] = { false };
        float s = 0.0f;
        for (int i = 0; i < 8; i++) {
            check(idx[i] >= 0 && idx[i] < 8, "k == n index in range");
            check(!seen[idx[i]], "k == n selects each expert once");
            seen[idx[i]] = true;
            s += w[i];
        }
        check(fabsf(s - 1.0f) < 1e-6f, "k == n weights renormalise to 1");
    }

    /* The shared-expert gate is a sigmoid. */
    check(fabsf(ds4_test_qwen35_shared_expert_gate(0.0f) - 0.5f) < 1e-6f,
          "shared expert gate at 0 is 0.5");
    check(ds4_test_qwen35_shared_expert_gate(20.0f) > 0.99f,
          "shared expert gate saturates high");
    check(ds4_test_qwen35_shared_expert_gate(-20.0f) < 0.01f,
          "shared expert gate saturates low");
}

int main(int argc, char **argv) {
    /* Render one turn and print it verbatim, so an external harness can diff it
     * against the GGUF's own Jinja template.  Used by msgcmp.py. */
    if (argc == 4 && !strcmp(argv[1], "--render")) {
        char out[8192];
        ds4_test_qwen35_render_turn(argv[2], argv[3], out, sizeof(out));
        fputs(out, stdout);
        return 0;
    }

    /* Route one logit vector and print "index weight" pairs, so an external
     * harness can diff it against a reference router. */
    if (argc >= 4 && !strcmp(argv[1], "--route")) {
        const uint32_t top_k = (uint32_t)strtoul(argv[2], NULL, 10);
        const uint32_t n = (uint32_t)(argc - 3);
        if (n == 0 || n > 384u) { fprintf(stderr, "bad expert count\n"); return 2; }
        float logits[384];
        for (uint32_t i = 0; i < n; i++) logits[i] = strtof(argv[3 + i], NULL);
        uint32_t idx[384];
        float w[384];
        const uint32_t got = ds4_test_qwen35_moe_route(logits, n, top_k, idx, w);
        for (uint32_t i = 0; i < got; i++) printf("%u %.9g\n", idx[i], (double)w[i]);
        return 0;
    }

    test_layer_typing();
    test_unicode_categories();
    test_piece_splitting();
    test_validation_coverage();
    test_quant_gap_report();
    test_chat_rendering();
    test_turn_rendering();
    test_moe_routing();

    if (failures) {
        printf("\n%d qwen35 check(s) failed\n", failures);
        return 1;
    }
    printf("qwen35 loader/tokenizer checks passed\n");
    return 0;
}
