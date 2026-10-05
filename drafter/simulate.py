#!/usr/bin/env python3
"""Simulate greedy verification against recorded target continuations."""

import argparse
import json
from pathlib import Path

import torch

from data import Shard
from model import DFlash2
from train import draft_paths, seq_forward

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def simulate(model, embd, ex, device, n_max=7):
    """Returns (tokens produced by verify steps, verify steps). The prefill token is excluded."""
    toks = ex["tokens"]
    n_prompt = ex["meta"]["n_prompt"]
    L = len(toks)
    if L - n_prompt < 2:
        return 0, 0
    B = n_max + 1
    anchors = torch.arange(n_prompt, L - 1)
    paths = []
    for i in range(0, len(anchors), 256):
        a = anchors[i:i + 256]
        hidden, *_ = seq_forward(model, embd, ex, a, B, device, ckpt=False)
        path, _ = draft_paths(model, hidden, toks.to(device)[a.to(device)])
        paths.append(path.cpu())
    paths = torch.cat(paths)
    pos, steps, gen = n_prompt, 0, 0
    while pos < L - 1:
        d = paths[pos - n_prompt]
        m = 0
        while m < len(d) and pos + 1 + m < L and d[m].item() == toks[pos + 1 + m].item():
            m += 1
        inc = min(m + 1, L - 1 - pos)
        steps += 1
        gen += inc
        pos += inc
    return gen, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--draft", default=str(ROOT / "models/dflash2-hf"))
    ap.add_argument("--n-max", type=int, default=7)
    a = ap.parse_args()
    device = "cuda"
    lm_head = torch.load(ROOT / "models/cache/output.weight.pt")
    embd = torch.load(ROOT / "models/cache/token_embd.weight.pt")
    model = DFlash2(a.draft, lm_head, rank=0).to(device).eval()
    sh = Shard(a.shard)
    G = S = 0
    for i in range(len(sh)):
        ex = sh.get(i)
        g, s = simulate(model, embd, ex, device, a.n_max)
        if not s:
            print(json.dumps({"id": ex["meta"]["id"], "skipped": "fewer than 2 response tokens"}), flush=True)
            continue
        G += g; S += s
        print(json.dumps({"id": ex["meta"]["id"], "tok_per_step": g / s, "gen": g, "steps": s}), flush=True)
    if not S:
        raise SystemExit("no sequence with at least 2 response tokens")
    print("TOTAL tok/step", G / S)


if __name__ == "__main__":
    main()
