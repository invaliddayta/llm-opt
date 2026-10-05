# Status - 2026-10-05

## Setup

- Target: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf` from
  `HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF` @ `993a5971`, KV cache q4_0
  (K and V), 98304 context, BF16 mmproj loaded.
- Draft: stock z-lab DFlash2, `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` from
  `z-lab/Qwen3.8-27B-DFlash2-GGUF` @ `2d9571f8`, f16 KV, `--spec-type draft-dflash`, n_max 7,
  p_min 0. No custom-trained drafter.
- Exact files and checksums: [README, Model](../README.md#model).
- GPU: one RTX 3090 (sm_86, 24 GB) over USB4. llama.cpp `8df332de1` + fork branch `opt/main`
  with `GGML_CUDA_FATTN_Q4_MMA=1 LLAMA_GPU_SAMPLING=1 LLAMA_DFLASH_GPU_FEATURES=1`.
- 98K rather than 196K context: at 196K the server could not reload after idle sleep while
  another process held ~2 GiB of GPU memory. 98K halves the KV cache and the sleep snapshot.

## Decode speed (task replay)

`bench/r20_replay.py`: three tasks as single direct requests, thinking on (explanation and
story: T=0.4, top-p 0.95, top-k 20, seed 1234; python100: T=0 with an 11-tool catalog).
Quiet GPU. Output hashes are identical across all bit-exact fork builds:
explanation `5d5e2fcd546f898d`, python100 `afe9e5cc5ef99d54`, story `36644797916dc4c3`.

| Build | explanation | python100 | story | weighted |
| --- | --- | --- | --- | --- |
| upstream `8df332de1` (no fork changes) | 53.3 ms/step, 108.2 tok/s | 55.1 ms/step, 96.2 tok/s | 53.8 ms/step, 72.8 tok/s | 90.5 tok/s |
| fork, same session as upstream | 32.7 ms/step, 150.1 tok/s | 33.4 ms/step, 163.5 tok/s | 33.3 ms/step, 129.8 tok/s | 143.7 tok/s |
| fork, 2026-10-04 | 33.0 ms/step | 33.7 ms/step | 33.7 ms/step | 141.6 tok/s |
| fork, 2026-10-05 | 32.4 ms/step, 151.4 tok/s | 33.3 ms/step, 164.0 tok/s | 33.3 ms/step, 129.8 tok/s | **144.1 tok/s** |

Same server flags for all rows (`bench/ab_bins.sh`, upstream and fork back to back on
2026-10-05; the fork measured 32.7/33.4/33.3 ms/step in that session). Upstream samples with a
different RNG, so its outputs and lengths differ and tok/s is only indicative there; ms per
verify step is the like-for-like number: **~1.6x faster** per step.

Tokens per step (4.3-5.5 here) depend on the content: draft acceptance is high on long prose and
code, ~3 tokens/step on short mixed chat. Another process using ~14% of the GPU raises the step
time from ~33 to ~40 ms.

### 2026-10-05: two bit-exact MMSQ changes

- 6-CTAs/SM code for grids that fill 8 CTAs/SM badly (attention q -19%, GDN gate -11%).
- Fused residual add + RMS norm + norm weight + MMSQ activation quantization (127 per step).
  The activation format now stores shift bits per token instead of per token pair.

Step breakdown, dropped ideas and next candidates: [design/experiments.md](design/experiments.md).

## Opt-in features (measured 2026-10-04, 98K context)

The baseline here is the fork with the three opt-ins off (MMSQ etc. still on).

Plain chat, `bench/bench.py`, 18 prompts x 768 tokens, sampled, thinking on:

| Config | tok/s | ms/step | tokens/step |
| --- | --- | --- | --- |
| opt-ins off | 79.3 | 36.6 | 2.91 |
| Q4 MMA attention only | 78.3 | 37.1 | 2.91 |
| all three | 80.3 | 36.9 | 2.96 |

Agent turn, `bench/agent_bench.py`, 11-tool catalog, thinking on, 2 x 1500 tokens:

| Context | opt-ins off | all three |
| --- | --- | --- |
| 2.6K | 72.8 | 81.8 |
| 30K | 64.3 | 65.1 |

Plain chat changes are within noise. The gain is on tool-carrying requests, where upstream
falls back to CPU sampling (logits copied to the host every step). Q4 MMA attention targets long
KV: 95.8 -> 109.7 tok/s at ~91K tokens. Two-run points vary by +-5 tok/s.

## Correctness and stability

- `flash_attn_ext_vec` shared-memory race (upstream): fixed, racecheck 2,222,080 -> 0 hazards,
  3982/3982 `FLASH_ATTN_EXT` tests.
- GPU sampling never rejects requests: penalties, logprobs, reasoning budget and unsupported
  grammars use standard sampling for that request.
- Fixed a server abort (`GGML_ASSERT(!gsmpl->grmr ...)`) when a grammar request met
  `--backend-sampling`: the server now checks `common_sampler_backend_sampling()`.
- DFlash2 GPU selector setup follows the configured mode, so it survives a fallback request
  followed by sleep/wake.
- `bench/compat_check.py` 9/9; `test-backend-ops` passes for every touched op.

## Known limits

- Logprobs for draft-accepted tokens report a placeholder probability of 1.0 without
  alternatives (same as upstream at this revision).
- Token embedding stays on the CPU: moving it to the GPU gave identical output and no speed
  gain (140.6 vs 141.2/141.6 tok/s) for +0.7 GiB VRAM.
