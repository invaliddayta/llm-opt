# shellcheck shell=bash
# Shared test-server start/wait/cleanup for the bench wrappers (sourced from the workbench root).
PORT="${PORT:-8181}"
export PORT
URL="http://127.0.0.1:$PORT"
server_pid=
mkdir -p runs

stop_server() {
    [[ -n $server_pid ]] || return 0
    kill -- "-$server_pid" 2> /dev/null || kill "$server_pid" 2> /dev/null || true
    wait "$server_pid" 2> /dev/null || true
    server_pid=
}
trap stop_server EXIT
trap 'exit 1' INT TERM HUP

# start_server LOG CMD...: runs CMD in its own process group and waits for /health.
start_server() {
    local log=$1
    shift
    if curl -s -o /dev/null --max-time 2 "$URL/health"; then
        echo "something already answers on port $PORT; stop it or set PORT" >&2
        exit 1
    fi
    setsid "$@" > "$log" 2>&1 &
    server_pid=$!
    for _ in $(seq "${READY_TIMEOUT:-600}"); do
        kill -0 "$server_pid" 2> /dev/null || break
        curl -fsS -o /dev/null --max-time 2 "$URL/health" 2> /dev/null && return 0
        sleep 1
    done
    echo "server did not become ready (see $log)" >&2
    exit 1
}
