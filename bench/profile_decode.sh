#!/usr/bin/env bash
# Nsight Systems trace of steady-state decode on a test server (production must be stopped).
# Usage (inside `nix develop`): bench/profile_decode.sh NAME [extra llama-server args...]
# Opt-in env vars as for serve_test.sh. Writes runs/NAME.nsys-rep.
set -uo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
session="prof-$$"
nsys launch --session-new="$session" --trace=cuda,nvtx,osrt --cuda-graph-trace="${GRAPH_TRACE:-node}" \
    bench/serve_test.sh "$@" > "runs/$name.server.log" 2>&1 &
pid=$!
for _ in $(seq 600); do curl -sf localhost:8181/health > /dev/null && break; kill -0 $pid 2> /dev/null || break; sleep 1; done
.venv/bin/python bench/r20_replay.py --url http://127.0.0.1:8181 --tasks explanation > "runs/$name.replay.log" 2>&1 &
replay=$!
sleep "${PROFILE_DELAY:-12}"
nsys start --session="$session" --output="runs/$name" --force-overwrite=true
sleep "${PROFILE_SECONDS:-6}"
nsys stop --session="$session"
wait $replay
cat "runs/$name.replay.log"
kill $pid 2> /dev/null; wait $pid 2> /dev/null
ls -la "runs/$name.nsys-rep"
