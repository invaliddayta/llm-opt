#!/usr/bin/env python3
"""Fine-tune the DFlash2 drafter on the deployed target's own outputs and features.

Objective per anchor a (block = [tok[a], MASK x (B-1)], position k predicts tok[a+k]):
  * unary loss: CE of the drafter distribution against the target's *sampling*
    distribution (top-k 20 / top-p 0.95 truncation of the dumped top-32 logits at
    position a+k-1), weighted by exp(-(k-1)/gamma);
  * selector loss: CE over the drafter's own top-16 candidates for the actual
    token, with the actual previous token as predecessor (teacher forcing).

Checkpoint selection uses the mean leading-match length of the greedy selector
path against the recorded (sampled) continuation. For a deterministic draft and
lossless speculative sampling, P(accept d_1..d_k) = P(target samples d_1..d_k),
so this is an unbiased estimate of the accepted length per step.
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.checkpoint import checkpoint

from data import Shard
from model import DFlash2, build_attn_mask

ROOT = Path(__file__).resolve().parent.parent


def target_dist(topk_ids, topk_lg, top_k=20, top_p=0.95):
    """Truncated target sampling distribution from dumped top-32 logits. Returns (ids [N,20], probs [N,20])."""
    ids = topk_ids[:, :top_k]
    p = torch.softmax(topk_lg[:, :top_k].float(), -1)
    cum = p.cumsum(-1)
    keep = (cum - p) < top_p  # smallest prefix reaching top_p
    p = p * keep
    return ids, p / p.sum(-1, keepdim=True)


def valid_anchors(loss_mask):
    v = (loss_mask[:-1] & loss_mask[1:]).nonzero().flatten()
    return v[v >= 1]


def seq_forward(model, embd, ex, anchors, B, device, ckpt=True):
    toks = ex["tokens"].to(device)
    L = toks.shape[0]
    feat = ex["feat"].to(device, torch.bfloat16)
    anchors = anchors.to(device)
    A = anchors.shape[0]
    pos = torch.arange(L, device=device)
    ctx_kv = model.context_kv(feat, pos)

    blk_ids = torch.full((A, B), model.mask_id, dtype=torch.long, device=device)
    blk_ids[:, 0] = toks[anchors]
    block_emb = embd[blk_ids.flatten().cpu()].to(device)
    mask = build_attn_mask(L, anchors, B, model.window, device)
    hidden = model.blocks_forward(ctx_kv, block_emb, anchors, mask, B, ckpt=ckpt)
    hidden = hidden.view(A, B, -1)[:, 1:]  # predictions for positions 1..B-1

    k = torch.arange(1, B, device=device)
    lab_pos = anchors[:, None] + k[None, :]
    in_seq = lab_pos < L
    lab_pos_c = lab_pos.clamp(max=L - 1)
    lm = ex["loss_mask"].to(device)[lab_pos_c] & in_seq
    labels = toks[lab_pos_c]
    pred = torch.cat([toks[anchors][:, None], labels[:, :-1]], 1)
    rows = (lab_pos_c - 1).cpu().flatten()
    t_ids, t_p = target_dist(ex["topk_ids"][rows].to(device), ex["topk_lg"][rows].to(device))
    return hidden, labels, pred, lm, t_ids, t_p


def head_terms(model, hidden, labels, pred, lm, t_ids, t_p, gamma, sel_alpha):
    A, K1, C = hidden.shape
    w_pos = torch.exp(-(torch.arange(K1, device=hidden.device).float()) / gamma)
    tot = {"unary": 0.0, "sel": 0.0, "w": 0.0, "wsel": 0.0}
    loss = 0.0
    chunk = 48

    def chunk_fn(h, lab, prd, m, tid, tp):
        logits = F.linear(h.reshape(-1, C), model.lm_head).float()
        logq = torch.log_softmax(logits, -1)
        soft = -(tp * logq.gather(1, tid)).sum(-1)
        wts = (m.float() * w_pos[None, :]).reshape(-1)
        u = (soft * wts).sum()
        un, cand = logits.topk(model.top_k, -1)
        labf = lab.reshape(-1)
        hit = cand.eq(labf[:, None])
        covered = hit.any(-1)
        tgt = hit.long().argmax(-1)
        sc = model.selector_scores(h.reshape(-1, C), cand, un, prd.reshape(-1))
        sce = F.cross_entropy(sc, tgt, reduction="none")
        wsel = wts * covered.float()
        return u, (sce * wsel).sum(), wts.sum(), wsel.sum()

    for i in range(0, A, chunk):
        sl = slice(i, i + chunk)
        tid = t_ids.view(A, K1, -1)[sl].reshape(-1, t_ids.shape[-1])
        tp = t_p.view(A, K1, -1)[sl].reshape(-1, t_p.shape[-1])
        u, s, w, ws = checkpoint(chunk_fn, hidden[sl], labels[sl], pred[sl], lm[sl], tid, tp, use_reentrant=False)
        loss = loss + u + sel_alpha * s
        tot["unary"] += u.item(); tot["sel"] += s.item(); tot["w"] += w.item(); tot["wsel"] += ws.item()
    return loss, tot


@torch.no_grad()
def draft_paths(model, hidden, prev):
    """Greedy selector walk. hidden [A, K1, C], prev [A] anchor tokens -> path [A, K1]."""
    A, K1, C = hidden.shape
    logits = F.linear(hidden.reshape(-1, C), model.lm_head).float()
    un, cand = logits.topk(model.top_k, -1)
    un = un.view(A, K1, -1); cand = cand.view(A, K1, -1)
    path = []
    for k in range(K1):
        sc = model.selector_scores(hidden[:, k], cand[:, k], un[:, k], prev)
        prev = cand[:, k].gather(1, sc.argmax(-1, keepdim=True))[:, 0]
        path.append(prev)
    return torch.stack(path, 1), cand[..., 0]


@torch.no_grad()
def evaluate(model, embd, shards, items, device, B=8, max_items=80):
    """Mean accepted tokens per verify step (1 + leading matches vs the sampled continuation)
    at every 3rd response anchor, for the selector path and for plain unary argmax."""
    model.eval()
    sel_sum, un_sum, n = 0.0, 0.0, 0
    reach = torch.zeros(B - 1)
    for si, i in items[:max_items]:
        ex = shards[si].get(i)
        va = valid_anchors(ex["loss_mask"])[::3]
        for j in range(0, len(va), 256):
            a = va[j:j + 256]
            hidden, labels, pred, lm, _, _ = seq_forward(model, embd, ex, a, B, device, ckpt=False)
            path, top1 = draft_paths(model, hidden, ex["tokens"].to(device)[a.to(device)])
            m_sel = torch.cumprod(((path == labels) & lm).float(), 1)
            m_un = torch.cumprod(((top1 == labels) & lm).float(), 1)
            sel_sum += (m_sel.sum(1) + 1).sum().item()
            un_sum += (m_un.sum(1) + 1).sum().item()
            reach += m_sel.sum(0).cpu()
            n += a.shape[0]
    model.train()
    return {"accept_len": sel_sum / max(n, 1), "accept_len_unary": un_sum / max(n, 1), "anchors": n,
            "reach": [round(x, 4) for x in (reach / max(n, 1)).tolist()]}


def split_by_prompt(shards, eval_frac, seed):
    """Group sequences by prompt token hash so identical prompts never straddle train/eval."""
    groups = {}
    for si, s in enumerate(shards):
        for i in range(len(s)):
            npr = s.meta[i]["n_prompt"]
            h = hashlib.sha1(s.tokens[i][:npr].tobytes()).hexdigest()
            groups.setdefault(h, []).append((si, i))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_eval = max(1, int(len(keys) * eval_frac))
    ev = [x for k in keys[:n_eval] for x in groups[k]]
    tr = [x for k in keys[n_eval:] for x in groups[k]]
    return tr, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--init", default=str(ROOT / "models/dflash2-hf"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-small", type=float, default=5e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--anchors", type=int, default=256)
    ap.add_argument("--blocks", default="8,8,6,4", help="runtime block lengths sampled per sequence")
    ap.add_argument("--gamma", type=float, default=5.0)
    ap.add_argument("--sel-alpha", type=float, default=0.5)
    ap.add_argument("--eval-frac", type=float, default=0.03)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-max", type=int, default=80)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = [int(x) for x in a.blocks.split(",")]

    lm_head = torch.load(ROOT / "models/cache/output.weight.pt")
    embd = torch.load(ROOT / "models/cache/token_embd.weight.pt")  # CPU bf16
    model = DFlash2(a.init, lm_head, rank=a.rank, alpha=float(a.rank)).to(device)
    del lm_head

    shards = [Shard(p) for p in a.shards]
    train_items, eval_items = split_by_prompt(shards, a.eval_frac, a.seed)
    print(f"train seqs {len(train_items)} eval seqs {len(eval_items)}", flush=True)
    if not eval_items or (not a.eval_only and not train_items):
        raise SystemExit("not enough data for a train/eval split")

    ev = evaluate(model, embd, shards, eval_items, device, max_items=a.eval_max)
    print("eval@0", json.dumps(ev), flush=True)
    log = open(out / "log.jsonl", "a")
    log.write(json.dumps({"step": 0, "eval": ev}) + "\n")
    if a.eval_only:
        return

    big, small = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (big if "lora_" in n else small).append(p)
    opt = torch.optim.AdamW([{"params": big, "lr": a.lr}, {"params": small, "lr": a.lr_small}], weight_decay=0.0, betas=(0.9, 0.99))
    total_steps = int(len(train_items) * a.epochs / a.accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    if total_steps < 1:
        raise SystemExit("no optimizer steps; add data or epochs")
    warm = max(5, total_steps // 30)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"trainable lora {sum(p.numel() for p in big)/1e6:.1f}M small {sum(p.numel() for p in small)/1e6:.1f}M steps {total_steps}", flush=True)

    order = []
    while len(order) < total_steps * a.accum:
        o = train_items[:]
        rng.shuffle(o)
        order += o
    agg = {"unary": 0.0, "sel": 0.0, "w": 0.0, "wsel": 0.0}
    best = ev["accept_len"]
    step, t0 = 0, time.time()
    for it, (si, i) in enumerate(order[: total_steps * a.accum]):
        ex = shards[si].get(i)
        va = valid_anchors(ex["loss_mask"])
        if len(va) > 0:
            if len(va) > a.anchors:
                va = va[torch.randperm(len(va))[: a.anchors]].sort().values
            B = rng.choice(blocks)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden, labels, pred, lm, t_ids, t_p = seq_forward(model, embd, ex, va, B, device)
                loss, tot = head_terms(model, hidden, labels, pred, lm, t_ids, t_p, a.gamma, a.sel_alpha)
            (loss / max(tot["w"], 1.0) / a.accum).backward()
            for kk in agg:
                agg[kk] += tot[kk]
        if (it + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(big + small, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                el = time.time() - t0
                msg = {"step": step, "unary": agg["unary"] / max(agg["w"], 1), "sel": agg["sel"] / max(agg["wsel"], 1),
                       "sel_cov": agg["wsel"] / max(agg["w"], 1), "lr": sched.get_last_lr()[0], "s_per_step": el / step,
                       "mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                print(json.dumps(msg), flush=True)
                log.write(json.dumps(msg) + "\n"); log.flush()
                agg = {k: 0.0 for k in agg}
            if step % a.eval_every == 0 or step == total_steps:
                ev = evaluate(model, embd, shards, eval_items, device, max_items=a.eval_max)
                print(f"eval@{step}", json.dumps(ev), flush=True)
                log.write(json.dumps({"step": step, "eval": ev}) + "\n"); log.flush()
                if ev["accept_len"] > best:
                    best = ev["accept_len"]
                    save_file(model.export_state_dict(), str(out / "model.safetensors"))
                    (out / "config.json").write_text((Path(a.init) / "config.json").read_text())
                    print("saved best", best, flush=True)


if __name__ == "__main__":
    main()
