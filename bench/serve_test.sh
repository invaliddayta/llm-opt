#!/usr/bin/env bash
# Start a test llama-server on port 8181 with the production server flags.
# Opt-ins come from the environment, e.g.
#   GGML_CUDA_FATTN_Q4_MMA=1 LLAMA_GPU_SAMPLING=1 LLAMA_DFLASH_GPU_FEATURES=1 bench/serve_test.sh
# Then run bench/agent_bench.py or bench/compat_check.py --url http://127.0.0.1:8181. Ctrl-C stops it.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${BIN:-llama.cpp/build/bin/llama-server}"
CTX="${CTX:-98304}"
store() { find /nix/store -maxdepth 1 -name "*-$1" -print -quit; }
MODEL="${MODEL:-$(store Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf)}"
DRAFT="${DRAFT:-$(store Qwen3.8-27B-DFlash2-Q4_K_M.gguf)}"
MMPROJ="${MMPROJ:-$(store mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf)}"
TEMPLATE="${TEMPLATE:-models/chat-template.jinja}"
for f in "$BIN" "$MODEL" "$DRAFT" "$MMPROJ" "$TEMPLATE"; do [[ -e "$f" ]] || { echo "missing: $f" >&2; exit 1; }; done

export LD_LIBRARY_PATH="$PWD/driver-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$BIN" -m "$MODEL" --alias local-model --host 127.0.0.1 --port 8181 \
    --ctx-size "$CTX" --parallel 1 --gpu-layers 999 --split-mode none --flash-attn on \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --spec-draft-model "$DRAFT" --spec-draft-device CUDA0 --spec-draft-ngl 999 --spec-draft-poll 1 \
    --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0 \
    --backend-sampling --threads 12 --threads-batch 16 --batch-size 2048 --ubatch-size 512 \
    --jinja --chat-template-file "$TEMPLATE" --reasoning-format deepseek --mmproj "$MMPROJ" --no-webui --poll 10 "$@"
