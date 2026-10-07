# drafter: DFlash2 fine-tuning (unfinished)

**Status: tried, no gain, not used.** Every result in this repo uses z-lab's stock
`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`. This directory holds the pipeline for adapting that drafter
to the deployed target's own outputs.

Result (2026-10-07, 2.1M tokens, 8% of prompts held out): stock 4.309 expected accepted tokens
per step; best fine-tune 4.315 (+0.1%, noise), and the default learning rates made it worse.
Details: [experiments.md](../docs/design/experiments.md#dropped). Training peaks at 12.1 GB
VRAM, so it cannot run next to the 22 GB server.

The idea: DFlash2 was trained against the original Qwen3.8-27B. The deployed target is a
fine-tuned IQ4_XS variant, so a drafter fitted to its actual sampling distribution should get
more drafted tokens accepted per verify step.

| File | Step |
| --- | --- |
| `prep_prompts.py` | Prompt mix from public datasets (UltraChat, Magicoder, OpenCodeInstruct, NuminaMath, GSM8K, Hermes function calling) |
| `gen.py` | Generates continuations with the deployed target through llama-server |
| `data.py` | Reads the shards written by `llama-dflash-dump` (fork `examples/dflash-dump`): target features, top-32 logits, token ids |
| `model.py` | PyTorch DFlash2 drafter (loads the z-lab HF weights) |
| `parity.py` | Checks the PyTorch drafter against llama.cpp (`llama-dflash-parity`) at the same anchors |
| `train.py` | LoRA fine-tune: CE against the target's truncated sampling distribution per block position, plus a selector loss; checkpoints chosen by expected accepted length |
| `overfit_check.py`, `simulate.py` | Sanity checks and acceptance simulation |
| `export_gguf.py` | Checkpoint to GGUF: overwrites the tensors of z-lab's BF16 GGUF in place (`--check` reproduces it exactly from the stock weights); then `llama-quantize ... Q4_K_M` reproduces z-lab's Q4_K_M byte for byte |
| `../bench/ab_drafter.sh` | Interleaved end-to-end A/B of two draft GGUFs |

A trained drafter would only be published (as a GGUF release) if it beats the stock drafter on
the task replay with the same target and settings.
