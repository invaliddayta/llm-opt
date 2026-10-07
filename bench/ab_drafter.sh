#!/usr/bin/env bash
# A/B two draft GGUFs (same target, flags and build), interleaved A B A B.
# Each round: R20 replay + agent benchmark. Production must be stopped (GPU memory).
# Usage: bench/ab_drafter.sh STOCK.gguf NEW.gguf [rounds]   (opt-in env vars as for serve_test.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
(($# >= 2)) || { echo "usage: bench/ab_drafter.sh STOCK.gguf NEW.gguf [rounds]" >&2; exit 2; }
# shellcheck source=bench/lib.sh
source bench/lib.sh
a=$1 b=$2 rounds=${3:-2}
for r in $(seq "$rounds"); do
    for side in A B; do
        [[ $side == A ]] && draft=$a || draft=$b
        name="drafter-ab-$side$r"
        DRAFT="$draft" start_server "runs/$name.server.log" bench/serve_test.sh
        echo "=== $name $draft"
        "${PYTHON:-python3}" bench/r20_replay.py --url "$URL" --out "runs/$name.json"
        "${PYTHON:-python3}" bench/agent_bench.py --url "$URL" --runs 1 --out "runs/$name.agent.json" | tail -4
        stop_server
    done
done
