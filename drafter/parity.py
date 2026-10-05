#!/usr/bin/env python3
"""Parity test: PyTorch drafter drafts vs llama.cpp's (llama-dflash-parity) at the same anchors.

usage: parity.py --shard data/shards/s0 --seq 3 --draft-hf models/dflash2-hf --draft-gguf models/.../BF16.gguf --n-max 7
"""

import argparse
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import torch

from data import Shard
from model import DFlash2
from train import draft_paths, seq_forward

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=str(ROOT / "data/shards/s0"))
    ap.add_argument("--seqs", default="3,11,42")
    ap.add_argument("--draft-hf", default=str(ROOT / "models/dflash2-hf"))
    ap.add_argument("--draft-gguf", default=str(ROOT / "models/dflash2-gguf/Qwen3.8-27B-DFlash2-BF16.gguf"))
    ap.add_argument("--n-max", type=int, default=7)
    a = ap.parse_args()

    sh = Shard(a.shard)
    B = a.n_max + 1
    tot_pos = tot_eq = tot_full = tot_blocks = 0
    jobs = []
    for si in [int(x) for x in a.seqs.split(",")]:
        ex = sh.get(si)
        toks = ex["tokens"].numpy()
        npr = ex["meta"]["n_prompt"]
        L = len(toks)
        cand = [npr, npr + 1, npr + 7, npr + 30, npr + 100, npr + 257, 2040, 2047, 2048, 2049, 2100, 2500, L - 20]
        anchors = sorted({c for c in cand if npr <= c < L - 1})
        fx = f"/tmp/opencode/parity_{si}.bin"
        with open(fx, "wb") as f:
            f.write(struct.pack("<I", L)); f.write(toks.astype(np.uint32).tobytes())
        env = dict(os.environ, LD_LIBRARY_PATH=str(ROOT / "driver-libs"))
        cmd = [str(ROOT / "llama.cpp/build/bin/llama-dflash-parity"), "-m", str(ROOT / "models/target-iq4xs.gguf"),
               "-md", a.draft_gguf, "--spec-type", "draft-dflash", "--spec-draft-n-max", str(a.n_max), "--spec-draft-p-min", "0",
               "-ngl", "999", "--spec-draft-ngl", "999", "-fa", "on", "-c", "8192", "-b", "4096", "-ub", "512",
               "--in", fx, "--anchors", ",".join(map(str, anchors))]
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        cpp = {}
        for line in out.stdout.strip().splitlines():
            parts = [int(x) for x in line.split()]
            cpp[parts[0]] = parts[1:]
        if not cpp:
            print(out.stderr[-3000:])
            raise SystemExit("parity tool produced no output")
        jobs.append((si, ex, toks, anchors, cpp))

    lm_head = torch.load(ROOT / "models/cache/output.weight.pt")
    embd = torch.load(ROOT / "models/cache/token_embd.weight.pt")
    model = DFlash2(a.draft_hf, lm_head, rank=0).cuda().eval()
    for si, ex, toks, anchors, cpp in jobs:
        at = torch.tensor(anchors)
        with torch.no_grad():
            hidden, *_ = seq_forward(model, embd, ex, at, B, "cuda", ckpt=False)
            path, _ = draft_paths(model, hidden, ex["tokens"].cuda()[at.cuda()])
        for j, an in enumerate(anchors):
            py = path[j].tolist()
            cc = cpp.get(an, [])
            k = 0
            while k < min(len(py), len(cc)) and py[k] == cc[k]:
                k += 1
            eq = sum(1 for x, y in zip(py, cc) if x == y)
            tot_pos += len(py); tot_eq += eq; tot_full += int(py == cc); tot_blocks += 1
            actual = toks[an + 1: an + B].tolist()
            print(f"seq {si} anchor {an:5d} prefix_agree {k}/{len(py)}  py={py}  cpp={cc}  actual={actual}")
    print(f"PARITY: identical blocks {tot_full}/{tot_blocks}, positions {tot_eq}/{tot_pos}")


if __name__ == "__main__":
    main()
