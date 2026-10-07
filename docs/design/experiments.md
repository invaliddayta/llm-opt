# Optimization experiments

What was tried, what it measured and why it was kept or dropped. Current state:
[../STATUS.md](../STATUS.md). Fork commits referenced below are on
[llama.cpp-opt](https://github.com/invaliddayta/llama.cpp-opt) `opt/main`.

## Where a decode step goes (2026-10-05, 98K context, quiet GPU)

nsys trace of an R20 replay (`bench/profile_decode.sh`, `bench/analyze_nsys.py`), about 33 ms per
speculative step:

| Part | ms/step | Notes |
| --- | --- | --- |
| Target verify graph (8 tokens, ~2190 kernels) | 27.9 | MMSQ mul_mat 20.4, gated_delta_net 1.7, rms_norm 0.9, quantize_x 0.8, flash attention 0.8, ~1000 other small kernels |
| Draft graph (DFlash2, ~280 kernels) | 3.7 | mul_mat 2.8 (Q6_K head 1.3), top_k_small 0.2 |
| Feature graph + single launches | 0.6 | |
| Host gaps | ~0.7 | the trace shows ~3 ms, but most of that is profiler overhead (see graph split) |

Weights read per step: ~16.6 GB (IQ4_XS 10.7, Q5_K 2.8, Q6_K head twice 2.1, draft 0.95). DRAM
read ceiling (`kernels/bw_probe.cu`) is ~840 GB/s, so pure weight streaming is ~19.8 ms. In
server, IQ4_XS big shapes run at ~750 GB/s, Q5_K ~800, Q6_K head ~830. A lab variant with
the arithmetic removed reaches 813 (IQ4) / 846 (Q5_K) GB/s: the math costs ~7% / ~4%.

## Kept

### MMSQ 6-CTAs/SM dispatch (`90f3a3a1f`)

`MINB=8` limits registers so 8 CTAs fit per SM; `MINB=4` gets ~165 registers and 6 CTAs/SM.
`kernels/mmsq_exact_sweep.sh` (KW=2, stages 2-4, MINB 3-8; all bit-exact) showed the
register-rich code is faster when the grid fits one 6/SM wave, or leaves only a small tail at
8/SM: attention q 12288x5120 -19%, GDN gate 6144x5120 -11%, K/V 1024x5120 -1%. Shapes that
fill 8/SM waves well (FFN, 10240, 5120x6144) stay on MINB=8. Same arithmetic, same output.

### Fused residual add + RMS norm + weight + activation quantization (`7b553f229`)

`ADD -> RMS_NORM -> MUL` at each layer boundary (127 per step) is one kernel that also writes
the MMSQ activations, so the following matmuls skip `quantize_x`. The first version paired two
tokens per CTA (the activation format packed the shift bits of two tokens in one word) and
was slower than the three kernels it replaced (10.9 vs 9.3 us). The format now stores 16 bits
per token; with one CTA per token the fused kernel is faster. Bit-exact: 192/192 `quantize_x`
cases and 384/384 mul_mat cases (4 types, 3 shapes, N=1-16, split on/off) identical to the old
format; R20 replay hashes unchanged. Opt out with `GGML_CUDA_MMSQ_FUSE_NORM=0`.

Both together: -0.3 ms/step end to end (explanation 33.0 -> 32.7, python100 33.7 -> 33.4).

### Earlier, deployed 2026-10-04

- Fused q4_0 KV loader in MMA flash attention (`GGML_CUDA_FATTN_Q4_MMA=1`): long replay at
  91K tokens 95.8 -> 109.7 tok/s; vector-kernel dispatch was 2.4-3.8x slower; a half2 dequant
  variant was slower than the default path. Design: [q4-mma-attention.md](q4-mma-attention.md).
- GPU sampling + tool grammar (`LLAMA_GPU_SAMPLING=1`): stock llama.cpp falls back to CPU
  sampling whenever the OpenCode tool grammar is attached, copying 7.9 MB of logits over USB4
  per step (~1.2 ms). Design: [gpu-grammar.md](gpu-grammar.md).
- GPU-resident DFlash target features (`LLAMA_DFLASH_GPU_FEATURES=1`): removes five 160 KB
  D2H and one 800 KB H2D copy per step. Design: [gpu-feature-bridge.md](gpu-feature-bridge.md).

## Dropped

- **Split CUDA graph launch.** `cudaGraphLaunch` of the 2189-node target graph takes ~1.6 ms of
  CPU, and the trace showed the first kernel starting 1.4 ms after the call. Capturing doubling
  chunks (32, 64, ...) made the first kernel start after 70 us, but end-to-end ms/step did not
  change: without the profiler the launch was already mostly hidden. Reverted.
- **Skipping GDN state snapshots.** `gated_delta_net` writes 8 per-token states per layer
  (~1.2 GB/step) for rollback. A timing-only switch (wrong output) saved 0.6-0.8 ms/step, an
  upper bound: a correct replay design must re-read inputs and state. The earlier lab
  (`kernels/gdn_replay_lab.cu`) measured replay 1703-1897 us vs 1783 us snapshots at 1-4
  accepted tokens. Not worth the rollback-contract redesign yet.
- **GPU token embedding** (`--override-tensor ^token_embd\.weight$=CUDA0`): identical output,
  140.6 vs 141.2/141.6 tok/s, +0.7 GiB VRAM.
- **MMSQ stages/KW changes on the big shapes.** All slower than KW=2/ST=2/MINB=8; KW changes
  also alter the summation order.
- **Separate activation quantization kernels** (warp per column pair): exact but no faster.
- **Quantized-KV vector attention for decode:** see above, 2.4-3.8x slower.
- **Fine-tuning the DFlash2 drafter on the deployed target (2026-10-07).** LoRA rank 128 on
  2 shards (1363 sequences, 2.1M tokens), 8% of prompts held out (45K draft positions), held-out
  eval every 25 steps, best-only save, early stop after 3 evals without gain. Stock DFlash2
  scores 4.309 expected accepted tokens per step. Default rates (LoRA 2e-4, selector/norms
  5e-5): 4.285 at step 25 and flat, worse during warmup, so a bad update direction, not
  overfitting. LoRA 2e-5, rest frozen: best 4.315 (+0.1%, within noise), then declining. Too
  small to measure end to end, so no A/B. 12.1 GB VRAM, ~3 s per step. The stock drafter
  already fits this target well; a gain would need far more data, not tuning against the same
  held-out set.

## Open ideas, by estimated value

1. MMSQ IQ4_XS math (lookup and shift work): up to ~0.75 ms if it reached the load-only rate.
2. More small-op fusion: SwiGLU + quantize before ffn_down (~0.3 ms), the GDN input chain
   (~10 small kernels per layer).
3. GDN replay instead of snapshots: < 0.8 ms, needs a new rollback contract (save the
   pre-verify state and q/k/v/g/beta, replay the accepted prefix, cover zero/full acceptance,
   sequence branching and sleep-cache serialization).
4. Flash attention at long context: ~0.8 ms at 15K tokens, grows with context.
5. `top_k_small_pass1` in the draft: 0.21 ms, ~5x its bandwidth bound.

## Method

- Labs first: `kernels/lab_all.cu` includes the production header, rotates >32 MiB of weights
  (`COLD_WEIGHTS=1`) and reports the minimum over 40 rounds, which holds up under other GPU load.
  `EXACT_BASELINE=1` compares whole outputs with the production configuration.
- End to end: `bench/ab_bins.sh BASE_DIR NEW_DIR` (two builds, interleaved) or
  `bench/ab_replay.sh` (flags/env). Output hashes must match for output-preserving changes; a
  ms/step difference below ~0.2 ms needs repeated rounds.
- Stop other GPU users first (one using ~14% of the GPU added ~20% step time) and any production server.
- Profiler numbers inflate host-side costs; confirm any host-side finding unprofiled.
