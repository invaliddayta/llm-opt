# Fused q4_0 KV loader for MMA flash attention

Deployed, opt-in with `GGML_CUDA_FATTN_Q4_MMA=1`.

## What it does

During decode with speculative verification, llama.cpp runs the 8-query MMA flash-attention
kernel. With a q4_0 KV cache, the default path first converts K/V to f16 and then runs the f16
kernel. The fused loader decodes q4_0 directly into the kernel's swizzled half2 shared-memory
tiles. Four values share one packed-pair and scale load. MMA, softmax, output reduction and
stream ordering are unchanged.

Scope (everything else stays on the original path): sm86, head size 256, 8 queries, one
sequence, 24 query / 4 KV heads, dense q4_0 K/V, complete aligned KV tiles. It is a
compile-time loader variant behind the existing MMA call path, so there's no new graph op and
no copy of the kernel. The new and old kernel pointers each get their own shared-memory
attribute cache; a review found a race in the first cache, which now uses a per-device
`std::once_flag`.

## Results

- Byte-exact against the default path: 5 contexts, then 42 graph/stride/offset/scale/mask
  cases. memcheck clean. Full CUDA backend tests 16512/16512 with the flag on.
- Attention kernel time, cold, 4K/32K/96K context: 71/283/789 us fused vs 82/401/1130 us default.
- Server: identical generations on all 9 A/B requests; a 91,200-token prompt decoded at
  95.8 -> 109.7 tok/s. Short-context gains are small.

## Rejected variants

- Forcing the q4_0 vector kernel: 199/1399/4339 us at 4K/32K/96K, 2.4-3.8x slower.
- half2 dequantization: exact, but 92/417/1193 us, slower than the default path.
- A first version that loaded each pair twice was slower than the default.

Fair linked baselines need shared cudart, hidden template visibility and the application's
fast-math flags.

## Side finding

`flash_attn_ext_vec` had a shared-memory write-after-read race on `KQ` between loop
iterations (also in upstream at this revision). Fixed with one `ggml_cuda_syncwarp()` at the end
of the loop body: racecheck went from 2,222,080 hazards to 0, and all 3982 `FLASH_ATTN_EXT`
tests pass.
