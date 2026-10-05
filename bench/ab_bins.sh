#!/usr/bin/env bash
# Interleaved A/B of two llama-server builds on the R20 replay (production must be stopped).
# Usage: bench/ab_bins.sh BASE_BIN_DIR NEW_BIN_DIR [rounds] [tasks...]
# Opt-in env vars as for serve_test.sh. Output hashes must match for output-preserving changes.
set -uo pipefail
cd "$(dirname "$0")/.."
base=$1 new=$2 rounds=${3:-2}; shift 3 2> /dev/null || shift $#
tasks=("${@:-explanation python100}")
for r in $(seq "$rounds"); do
    for arm in base new; do
        dir=$base; [[ $arm == new ]] && dir=$new
        BIN="$dir/llama-server" LD_LIBRARY_PATH="$dir" bench/serve_test.sh > "runs/ab-$arm.server.log" 2>&1 &
        pid=$!
        for _ in $(seq 300); do curl -sf localhost:8181/health > /dev/null && break; kill -0 $pid 2> /dev/null || break; sleep 1; done
        .venv/bin/python bench/r20_replay.py --url http://127.0.0.1:8181 --tasks ${tasks[@]} | grep ms/step | sed "s/^/$arm r$r /"
        kill $pid; wait $pid 2> /dev/null
    done
done
