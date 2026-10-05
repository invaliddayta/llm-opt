#!/usr/bin/env bash
# A/B server variants on the R20 replay. Production must be stopped (GPU memory).
# Usage: bench/ab_replay.sh NAME [extra llama-server args...]   (opt-in env vars as for serve_test.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
(($# >= 1)) || { echo "usage: bench/ab_replay.sh NAME [extra llama-server args...]" >&2; exit 2; }
# shellcheck source=bench/lib.sh
source bench/lib.sh
name=$1; shift
start_server "runs/$name.server.log" bench/serve_test.sh "$@"
echo "=== $name $*"
grep -E 'CUDA0 (model|KV|compute) buffer size' "runs/$name.server.log" | tail -4 || true
"${PYTHON:-python3}" bench/r20_replay.py --url "$URL" --out "runs/$name.json"
