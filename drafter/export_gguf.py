#!/usr/bin/env python3
"""Write a DFlash2 GGUF from a train.py checkpoint.

Copies the stock z-lab BF16 GGUF (metadata, tensor layout and types unchanged) and overwrites
every tensor's data in place with the checkpoint's values. Running it on the stock HF
checkpoint must reproduce the stock GGUF byte for byte (--check).
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
from gguf import GGUFReader  # noqa: E402

LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "attention_conv.base_kernel": "attn_conv_base",
    "attention_conv.kernel_projection.weight": "attn_conv_proj.weight",
    "mlp_conv.base_kernel": "ffn_conv_base",
    "mlp_conv.kernel_projection.weight": "ffn_conv_proj.weight",
}
TOP = {
    "fc.weight": "fc.weight",
    "hidden_norm.weight": "enc.output_norm.weight",
    "norm.weight": "output_norm.weight",
    "candidate_selector.predecessor_codebook": "selector_predecessor.weight",
    "candidate_selector.successor_codebook": "selector_successor.weight",
    "candidate_selector.hidden_projection.weight": "selector_hidden.weight",
}


def gguf_name(hf):
    if hf in TOP:
        return TOP[hf]
    _, i, rest = hf.split(".", 2)
    return f"blk.{i}.{LAYER[rest]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="directory with model.safetensors")
    ap.add_argument("--base", default=str(ROOT / "models/dflash2-gguf/Qwen3.8-27B-DFlash2-BF16.gguf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="compare with --base instead of writing")
    a = ap.parse_args()

    sd = {gguf_name(k): v for k, v in load_file(str(Path(a.ckpt) / "model.safetensors")).items()}
    reader = GGUFReader(a.base)
    names = {t.name for t in reader.tensors}
    if set(sd) != names:
        raise SystemExit(f"tensor set mismatch: missing {sorted(names - set(sd))} extra {sorted(set(sd) - names)}")
    if not a.check:
        shutil.copyfile(a.base, a.out)
    out = None if a.check else np.memmap(a.out, dtype=np.uint8, mode="r+")
    diff = 0
    for t in reader.tensors:
        v = sd[t.name]
        if t.tensor_type.name == "BF16":
            raw = v.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint8).reshape(-1)
        elif t.tensor_type.name == "F32":
            raw = v.float().contiguous().numpy().view(np.uint8).reshape(-1)
        else:
            raise SystemExit(f"{t.name}: unexpected type {t.tensor_type.name}")
        if raw.size != t.n_bytes:
            raise SystemExit(f"{t.name}: {raw.size} bytes, GGUF has {t.n_bytes}")
        if a.check:
            diff += int(not np.array_equal(raw, np.asarray(t.data).view(np.uint8).reshape(-1)))
        else:
            out[t.data_offset:t.data_offset + t.n_bytes] = raw
    if a.check:
        print(f"{len(reader.tensors)} tensors, {diff} differ")
        sys.exit(1 if diff else 0)
    out.flush()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
