# GPU-resident DFlash target features

Deployed, opt-in with `LLAMA_DFLASH_GPU_FEATURES=1`.

## Why

The DFlash2 drafter conditions on hidden states from five target layers. Upstream copies them
to the host after every verify step and back for the draft: five 160 KB D2H copies and one
800 KB H2D copy per step.

## Design

- The target context owns one contiguous F32 CUDA staging tensor (`hidden x n_extract`, capacity
  `n_batch`; 200 MiB at batch 2048). At the end of each target microbatch, the selected layers
  are copied into their row-interleaved slices with device-to-device `cudaMemcpy2DAsync`, before
  the scheduler can reuse the source buffers. Extraction order follows the draft metadata.
- Staging is invalidated at every target decode/encode start, on state loads and on extraction
  config changes. The row count is published only after all microbatches succeed.
- Draft injection: a graph parameter (`dflash_device_features`) keeps injection and noise
  graphs apart; the injection input is filled by a D2D copy on the draft stream after graph
  allocation. The producer context is synchronized before the consumer copies.
- Scope: single sequence, DFlash2 fully on one CUDA device. Anything else uses the original CPU
  path, which is unchanged when the flag is off.

## Validation

Exact feature/selector-lattice comparisons against the CPU path, prose/Python/tool/image
requests at production context, and a graph trace with zero feature D2H/H2D and zero raw-logit
D2H per steady step. A separate per-clone device-query fix removed ~1.8 ms of GPU idle per step.
