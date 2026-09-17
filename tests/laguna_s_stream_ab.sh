#!/bin/zsh
# A/B: resident vs streamed output must agree for Laguna S 2.1 (greedy decode).
#
# Phase A wired S 2.1's routed experts through the Metal streaming expert
# cache: admission at the streaming/prefill gates, the decode per-token cap
# raised to 16 (S selects 10 experts/token), and the Q3_K addr path enabled in
# batch prefill. As with XS, a broken streamed path can fail *plausibly*
# (coherent-but-wrong tokens), so "it generated something reasonable" proves
# nothing. This gate is the correctness evidence: identical greedy output,
# resident vs streamed, byte for byte.
#
# Cache size matters. 3200 is the populated size the probe measured
# (budget=3200 experts entries=3200, expert=3.87 MiB, the Q3_K slab class).
# 800 is too small for 10-of-256 experts across 27 Q3_K layers to show hits,
# so an undersized cache would not exercise the path this gate exists to test.
#
# Bytewise comparison follows the repo's existing convention for greedy
# regressions (CONTRIBUTING.md "Correctness Regression Tests" -- the
# --logprob-vectors test compares local token bytes directly).
#
# Usage: ./tests/laguna_s_stream_ab.sh [MODEL]
set -e

MODEL=${1:-$HOME/models/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf}
CACHE_EXPERTS=${S21_AB_CACHE_EXPERTS:-3200}
CTX=4096
NTOK=128

if [[ ! -f $MODEL ]]; then
  echo "S stream A/B: model not found: $MODEL" >&2
  exit 1
fi

# Each ds4 invocation maps tens of GiB. Two of these running at once (or
# alongside a full ./ds4_test, which maps ~82 GiB) will drive a 128 GiB
# machine into swap. Refuse rather than wedge the host.
if pgrep -x ds4 >/dev/null 2>&1 || pgrep -x ds4_test >/dev/null 2>&1; then
  echo "S stream A/B: another ds4/ds4_test is running; refusing to contend" >&2
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
  ERR=$(mktemp)
  B=$(./ds4 -m $MODEL --ssd-streaming --ssd-streaming-cache-experts $CACHE_EXPERTS \
        -c $CTX -p "$p" -n $NTOK --nothink --temp 0 2>"$ERR")
  HITS=$(grep -o 'hits=[0-9]* misses=[0-9]* hit_rate=[0-9.]*' "$ERR" | tail -1 || true)
  if [[ -n $HITS ]]; then
    echo "streamed cache ($CACHE_EXPERTS experts): $HITS"
  else
    echo "streamed cache ($CACHE_EXPERTS experts): no hits=/misses= line"
  fi
  rm -f "$ERR"
  if [[ "$A" != "$B" ]]; then
    echo "MISMATCH on prompt: $p"
    # Keep the evidence -- a mismatch is a streaming bug and the divergence
    # point is the first thing you need when chasing it.
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

echo "S stream A/B: OK (cache=$CACHE_EXPERTS experts, ctx=$CTX, n=$NTOK)"
