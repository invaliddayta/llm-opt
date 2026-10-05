#!/usr/bin/env bash
# Sweep output-preserving MMSQ launch knobs (stages, min CTAs per SM) at KW=2 for given shapes.
# Usage (inside `nix develop`): kernels/mmsq_exact_sweep.sh "6144x5120,5120x6144" "2 3 4" "3 4 6 8" [types...]
set -euo pipefail
cd "$(dirname "$0")"
shapes=$1 stages=$2 minbs=$3; shift 3
types=("${@:-iq4_xs q5_K}")
out=/tmp/opencode/mmsq
mkdir -p "$out"
for st in $stages; do
    for mb in $minbs; do
        bin="$out/sweep-st$st-mb$mb"
        nvcc -O3 -arch=sm_86 -std=c++17 -DKW_H=2 -DST_H="$st" -DMINB_H="$mb" lab_all.cu -o "$bin"
        exact=$(SHAPES=$shapes NS=8 EXACT_BASELINE=1 "$bin" check ${types[@]} | grep -c 'EXACT_BASELINE.*OK' || true)
        echo "== ST=$st MINB=$mb exact_ok=$exact"
        SHAPES=$shapes NS=8 COLD_WEIGHTS=1 "$bin" bench ${types[@]} | grep -v '^AGG'
    done
done
