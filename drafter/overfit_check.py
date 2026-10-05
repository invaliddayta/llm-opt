#!/usr/bin/env python3
"""Sanity check: repeated steps on 4 fixed sequences must drive their loss down."""
import random
from pathlib import Path

import torch

from data import Shard
from model import DFlash2
from train import head_terms, seq_forward, valid_anchors

ROOT = Path(__file__).resolve().parent.parent
torch.manual_seed(0)
lm_head = torch.load(ROOT / "models/cache/output.weight.pt")
embd = torch.load(ROOT / "models/cache/token_embd.weight.pt")
model = DFlash2(ROOT / "models/dflash2-hf", lm_head, rank=128, alpha=128.0).cuda()
sh = Shard(ROOT / "data/shards/s0")
exs = [sh.get(i) for i in (1, 2, 5, 9)]
anch = []
for ex in exs:
    va = valid_anchors(ex["loss_mask"])
    anch.append(va[torch.randperm(len(va))[:256]].sort().values)
big = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
small = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
opt = torch.optim.AdamW([{"params": big, "lr": 1e-4}, {"params": small, "lr": 2e-5}], weight_decay=0.0)


def run(train):
    tu = tw = ts = tws = 0.0
    for ex, a in zip(exs, anch):
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(train):
            h, lab, prd, lm, tid, tp = seq_forward(model, embd, ex, a, 8, "cuda", ckpt=train)
            loss, tot = head_terms(model, h, lab, prd, lm, tid, tp, 5.0, 0.5)
        if train:
            (loss / tot["w"] / 4).backward()
        tu += tot["unary"]; tw += tot["w"]; ts += tot["sel"]; tws += tot["wsel"]
    return tu / tw, ts / tws


print("before", run(False))
for s in range(30):
    u = run(True)
    gn = torch.nn.utils.clip_grad_norm_(big + small, 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if s % 5 == 0:
        print(s, "train", u, "gradnorm", float(gn))
print("after", run(False))
