#!/bin/zsh
# P18 mlock-cap repro: streamed S prefill must COMPLETE when locked memory is
# capped below static-weights + cache.
#
# The condition that fails at layer 21 on a machine whose mlock ceiling is
# ~4.6 GiB (P18, 2026-09-17). Pre-fix behaviour: `Laguna batch prefill failed
# in routed experts after 21/48 layers`, engine exit 1. Post-fix: the expert is
# served from a mapped model view and the prefill completes.
#
# Usage: ./tests/p18_mlock_cap_repro.sh [MODEL]
set -e

MODEL=${1:-$HOME/models/laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf}
CACHE_EXPERTS=${P18_CAP_CACHE_EXPERTS:-192}
CAP_KB=${P18_MLOCK_CAP_KB:-4823449}   # ~4.6 GiB: static 4.07 GiB locks, cache cannot
CTX=4096
NTOK=4

if [[ ! -f $MODEL ]]; then
  echo "p18 cap repro: model not found: $MODEL" >&2
  exit 1
fi

# One model-loading process at a time.
if pgrep -x ds4 >/dev/null 2>&1 || pgrep -x ds4_test >/dev/null 2>&1; then
  echo "p18 cap repro: another ds4/ds4_test is running; refusing to contend" >&2
  exit 1
fi

P="The user wants a careful reading of the workspace documentation before answering the question about the glossary. "
P="$P$P$P"

ERR=$(mktemp)
rc=0
(
  ulimit -l $CAP_KB
  ./ds4 --metal --ssd-streaming --ssd-streaming-cache-experts $CACHE_EXPERTS \
    --prefill-chunk 512 -m $MODEL -c $CTX --temp 0 --nothink -n $NTOK -p "$P"
) >/dev/null 2>"$ERR" || rc=$?

if [[ $rc -ne 0 ]] || grep -qa "failed in routed experts" "$ERR"; then
  echo "p18 cap repro: FAIL — prefill did not complete under the mlock cap (exit $rc)"
  grep -a -E "failed in routed experts|mlock|failed buffer" "$ERR" | tail -5
  echo "  stderr retained: $ERR"
  exit 1
fi

echo "p18 cap repro: PASS — prefill completed under the mlock cap (exit 0)"
rm -f "$ERR"
