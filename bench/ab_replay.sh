#!/usr/bin/env bash
# A/B server variants on the R20 replay. Production must be stopped (GPU memory).
# Usage: bench/ab_replay.sh NAME [extra llama-server args...]   (opt-in env vars as for serve_test.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
bench/serve_test.sh "$@" > "runs/$name.server.log" 2>&1 &
pid=$!
for _ in $(seq 600); do curl -sf localhost:8181/health > /dev/null && break; kill -0 $pid 2> /dev/null || break; sleep 1; done
echo "=== $name $*"
grep -E 'CUDA0 (model|KV|compute) buffer size' "runs/$name.server.log" | tail -4
.venv/bin/python bench/r20_replay.py --url http://127.0.0.1:8181 --out "runs/$name.json"
kill $pid; wait $pid 2> /dev/null
