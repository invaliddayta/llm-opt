#!/usr/bin/env bash
# Interleaved A/B of two llama-server builds on the R20 replay (production must be stopped).
# Usage: bench/ab_bins.sh BASE_BIN_DIR NEW_BIN_DIR [rounds] [tasks...]
# Opt-in env vars as for serve_test.sh. Output hashes must match for output-preserving changes.
set -euo pipefail
cd "$(dirname "$0")/.."
(($# >= 2)) || { echo "usage: bench/ab_bins.sh BASE_BIN_DIR NEW_BIN_DIR [rounds] [tasks...]" >&2; exit 2; }
# shellcheck source=bench/lib.sh
source bench/lib.sh
base=$1 new=$2 rounds=${3:-2}
shift $(($# < 3 ? $# : 3))
tasks=("$@")
((${#tasks[@]})) || tasks=(explanation python100)
for r in $(seq "$rounds"); do
    for arm in base new; do
        dir=$base; [[ $arm == new ]] && dir=$new
        start_server "runs/ab-$arm.server.log" env BIN="$dir/llama-server" LD_LIBRARY_PATH="$dir" bench/serve_test.sh
        "${PYTHON:-python3}" bench/r20_replay.py --url "$URL" --tasks "${tasks[@]}" | grep ms/step | sed "s/^/$arm r$r /"
        stop_server
    done
done
