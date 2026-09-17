#!/bin/zsh
# A/B: resident vs streamed output must agree for Laguna XS 2.1 (greedy decode).
#
# Task 6 wired XS 2.1's routed experts through the Metal streaming expert
# cache. Task 3 already showed this family fails *plausibly* -- its decode
# dispatch bug produced coherent-but-wrong tokens rather than a crash -- so
# "streaming generates something reasonable" proves nothing. This gate is what
# actually proves it: identical greedy output, resident vs streamed.
#
# The cache is deliberately undersized (800 experts, well under one layer's
# worth across several layers) to force heavy eviction. A correct
# implementation is bytewise-identical to resident anyway, because every cache
# miss falls back to the mapped-model path.
#
# Bytewise comparison follows the repo's existing convention for greedy
# regressions (CONTRIBUTING.md "Correctness Regression Tests" -- the
# --logprob-vectors test compares local token bytes directly).
#
# Usage: ./tests/xs21_stream_ab.sh [MODEL]
set -e

MODEL=${1:-gguf/Laguna-XS-2.1-Q4_K_M.gguf}
CACHE_EXPERTS=${XS21_AB_CACHE_EXPERTS:-800}
CTX=8192
NTOK=128

if [[ ! -f $MODEL ]]; then
  echo "xs21 stream A/B: model not found: $MODEL" >&2
  exit 1
fi

# Each ds4 invocation maps ~19 GiB. Two of these running at once (or alongside
# a full ./ds4_test, which maps ~82 GiB) will drive a 128 GiB machine into
# swap. Refuse rather than wedge the host -- this has bitten this branch once.
if pgrep -x ds4 >/dev/null 2>&1 || pgrep -x ds4_test >/dev/null 2>&1; then
  echo "xs21 stream A/B: another ds4/ds4_test is running; refusing to contend" >&2
  exit 1
fi

# A CPU `ds4` (left by `make cpu`, which check-engine runs) carries none of the
# Metal/Laguna symbols and would silently decode on the CPU, so an A/B run
# against it proves nothing. Refuse a non-Metal binary outright.
if ! nm ds4 2>/dev/null | grep -q ds4_gpu_laguna_qk_head_rms_norm_rope_tensor; then
  echo "xs21 stream A/B: ds4 is not a Metal build; run 'make ds4' in external/ds4" >&2
  exit 1
fi

PROMPTS=(
  "def fizzbuzz(n):"
  "<html><head><title>"
  "Explain HTTP caching briefly."
  "import asyncio"
)

FAILED=0
for p in $PROMPTS; do
  A=$(./ds4 -m $MODEL -c $CTX -p "$p" -n $NTOK --nothink --temp 0)
  B=$(./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $CACHE_EXPERTS \
        -c $CTX -p "$p" -n $NTOK --nothink --temp 0)
  if [[ "$A" != "$B" ]]; then
    echo "MISMATCH on prompt: $p"
    # Keep the evidence -- a mismatch is a Task 6 bug and the divergence point
    # is the first thing you need when chasing it.
    OUT=$(mktemp -d)
    print -r -- "$A" > $OUT/resident.txt
    print -r -- "$B" > $OUT/streamed.txt
    echo "  resident: $OUT/resident.txt"
    echo "  streamed: $OUT/streamed.txt"
    diff $OUT/resident.txt $OUT/streamed.txt | head -20 || true
    FAILED=1
    break
  fi
done

if (( FAILED )); then
  exit 1
fi

echo "xs21 stream A/B: OK"
