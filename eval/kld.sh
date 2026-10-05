#!/usr/bin/env bash
# KL divergence of small-ubatch paths (where mmsq/mmvq run) against the large-ubatch reference.
# usage: kld.sh [label=env ...]   e.g. kld.sh stock=GGML_CUDA_MMSQ=0 mmsq=GGML_CUDA_MMSQ=1
set -u
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=$PWD/driver-libs
P=(./llama.cpp/build/bin/llama-perplexity -m models/target-iq4xs.gguf -ngl 999 -fa on -ctk q4_0 -ctv q4_0 -f eval/kld.txt -c 512 --chunks 12)
REF=${REF:-/tmp/opencode/kld_ref.bin}
if [ ! -f $REF ]; then
  env ${REF_ENV:-} GGML_CUDA_MMSQ=0 "${P[@]}" -b 512 -ub 512 --kl-divergence-base $REF > /tmp/opencode/kld_ref.log 2>&1 || { tail -5 /tmp/opencode/kld_ref.log; exit 1; }
fi
for cfg in "$@"; do
  label=${cfg%%=*}; envs=${cfg#*=}
  env $envs "${P[@]}" -b 512 -ub ${UB:-8} --kl-divergence-base $REF --kl-divergence > /tmp/opencode/kld_$label.log 2>&1
  echo "== $label (ub=${UB:-8})"
  grep -E "Mean    KLD|Mean PPL\(Q\)|Same top|99.9%  KLD|Maximum KLD" /tmp/opencode/kld_$label.log
done
