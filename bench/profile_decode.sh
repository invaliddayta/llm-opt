#!/usr/bin/env bash
# Nsight Systems trace of steady-state decode on a test server (production must be stopped).
# Usage (inside `nix develop`): bench/profile_decode.sh NAME [extra llama-server args...]
# Opt-in env vars as for serve_test.sh. Writes runs/NAME.nsys-rep.
set -euo pipefail
cd "$(dirname "$0")/.."
(($# >= 1)) || { echo "usage: bench/profile_decode.sh NAME [extra llama-server args...]" >&2; exit 2; }
# shellcheck source=bench/lib.sh
source bench/lib.sh
name=$1; shift
session="prof-$$"
replay=
cleanup() {
    [[ -z $replay ]] || kill "$replay" 2> /dev/null || true
    nsys shutdown --kill sigterm --session="$session" > /dev/null 2>&1 || true
    stop_server
}
trap cleanup EXIT
start_server "runs/$name.server.log" nsys launch --session-new="$session" --trace=cuda,nvtx,osrt \
    --cuda-graph-trace="${GRAPH_TRACE:-node}" bench/serve_test.sh "$@"
"${PYTHON:-python3}" bench/r20_replay.py --url "$URL" --tasks explanation > "runs/$name.replay.log" 2>&1 &
replay=$!
sleep "${PROFILE_DELAY:-12}"
nsys start --session="$session" --output="runs/$name" --force-overwrite=true
sleep "${PROFILE_SECONDS:-6}"
nsys stop --session="$session"
wait "$replay"
replay=
cat "runs/$name.replay.log"
cleanup
ls -la "runs/$name.nsys-rep"
