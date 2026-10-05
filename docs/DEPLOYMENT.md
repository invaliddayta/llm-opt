# Changing, testing and shipping the runtime

## 1. Develop

```sh
cd llama.cpp                         # checkout of llama.cpp-opt
git switch opt/main                  # or branch off it
nix develop ..                       # CUDA toolkit, compute-sanitizer, cmake, ninja
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_WEBUI=OFF -DGGML_CUDA_FA_QUANTS="q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
ninja -C build test-backend-ops llama-server llama-bench
```

`opt/main` is upstream `8df332de1` (tag `upstream-base`) plus a linear series of private commits.
Keep it linear; add an `Assisted-by:` trailer when an assistant wrote the change.

## 2. Test

The test server needs ~20 GB of GPU memory, so stop other GPU users (and a production server on
the same GPU) first. Run the backend checks (`build/bin/...`, `compute-sanitizer`) from `llama.cpp/`
and the `bench/` commands from the workbench root.

- Kernel labs first: `kernels/lab_all.cu` (`EXACT_BASELINE=1` for whole-output comparison with
  the production configuration), see [design/experiments.md](design/experiments.md).
- Correctness of touched ops: `build/bin/test-backend-ops -o MUL_MAT -b CUDA0` (any `-o` op).
- Races: `compute-sanitizer --tool racecheck --racecheck-report hazard build/bin/test-backend-ops -o FLASH_ATTN_EXT -b CUDA0 -p "hsk=256.*nb=1,"`
  must report `0 hazards`. Use `--tool memcheck` for out-of-bounds checks.
- Server: `GGML_CUDA_FATTN_Q4_MMA=1 LLAMA_GPU_SAMPLING=1 LLAMA_DFLASH_GPU_FEATURES=1 bench/serve_test.sh`
  (port 8181), then `bench/compat_check.py --url http://127.0.0.1:8181` must print `9/9 passed`.
- Output-preserving changes: `bench/ab_bins.sh <copy of the old build/bin> llama.cpp/build/bin`
  must print the same `sha=` per task for both builds; compare `ms/step` over 2+ rounds.
- Speed: `bench/r20_replay.py --url http://127.0.0.1:8181`, `bench/agent_bench.py` (tool catalog,
  2.6K and 30K context) and `bench/bench.py` (plain chat). `bench.py` starts its own server on port
  8181, so stop the test server first; it needs `--name`, plus `--draft` with `--spec dflash`, e.g.
  `bench/bench.py --name dflash --spec dflash --draft models/...gguf`.
  Small differences (< ~0.2 ms/step, +-5 tok/s) need repeated runs.

`bench/serve_test.sh` expects the GGUFs in the Nix store or via `MODEL`, `DRAFT` and optionally
`MMPROJ` (no `--mmproj` without it), and a chat template in `models/chat-template.jinja` or `TEMPLATE`.
`PORT` (default 8181) changes the port; the wrapper scripts use the same `PORT`.

Runtime switches:

| Env var | Default | Effect |
| --- | --- | --- |
| `GGML_CUDA_FATTN_Q4_MMA=1` | off | Fused q4_0 KV loader in the 8-query MMA flash attention (sm86, head 256, 24/4 heads). |
| `LLAMA_GPU_SAMPLING=1` | off | Sampling and tool grammar on the GPU. Unsupported requests (penalties, logprobs, reasoning budget, regex triggers, non-lazy grammar prefill) fall back to standard sampling per request. |
| `LLAMA_DFLASH_GPU_FEATURES=1` | off | DFlash target features stay on the GPU. |
| `GGML_CUDA_MMSQ=0` | on | Disables the MMSQ small-batch GEMM (and its fusions). |
| `GGML_CUDA_MMSQ_FUSE_NORM=0` | on | Disables the fused add + RMS norm + quantization. |

## 3. Ship (author's setup)

Production is a Podman container built by a private Nix flake (`hermes`). It builds upstream
llama.cpp `8df332de1` with one patch generated from the fork:

```sh
cd llama.cpp
git diff upstream-base opt/main -- . ':!examples' ':!README.md' ':!media/llama-cpp-opt.svg' > ~/hermes/patches/llama-opt.patch
cd ~/hermes
git add patches/llama-opt.patch            # flakes only see tracked files
nix build .#llama-cuda --no-link           # compile check (~10 min)
nix run .#serve-up                         # rebuilds the image, recreates the container
python3 ~/github/llm-opt/bench/compat_check.py   # against production, port 8080
```

`examples/` (the `dflash-dump` tools for `drafter/`) and the fork README and banner stay out of the patch. The opt-ins are
`model.nix` settings (`cudaFattnQ4Mma`, `gpuSampling`, `dflashGpuFeatures`). Any change to the
build or settings changes the serve revision, which recreates the container and invalidates
the sleep cache.

Rollback: set the three opt-ins to `false` (same binary, upstream code paths) or check out the
previous hermes commit. Then push: `git push` here, `git -C llama.cpp push private opt/main`,
and the hermes repo.
