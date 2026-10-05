"""Shard I/O for drafter training data.

A shard is: <name>.jsonl (gen rows) + <name>.seqs.bin (tokens fed to the dump tool)
+ <name>.feat/.topk/.idx written by llama-dflash-dump.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch

N_FEAT = 5 * 5120
TOPK = 32


def write_seqs(gen_jsonl, out_bin, max_len=4096):
    rows = []
    with open(out_bin, "wb") as f:
        for line in open(gen_jsonl):
            r = json.loads(line)
            toks = (r["prompt_tokens"] + r["gen_tokens"])[:max_len]
            if len(r["gen_tokens"]) < 8:
                continue
            f.write(struct.pack("<I", len(toks)))
            f.write(np.asarray(toks, dtype=np.uint32).tobytes())
            rows.append({"id": r["id"], "n_prompt": len(r["prompt_tokens"]), "n": len(toks), "source": r["source"], "effort": r["effort"]})
    with open(str(out_bin) + ".meta.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


class Shard:
    def __init__(self, prefix):
        prefix = str(prefix)
        idx = np.fromfile(prefix + ".idx", dtype=np.dtype([("off", "<u8"), ("n", "<u4")]))
        self.off = idx["off"].astype(np.int64)
        self.n = idx["n"].astype(np.int64)
        total = int(self.off[-1] + self.n[-1])
        self.feat = np.memmap(prefix + ".feat", dtype=np.float16, mode="r", shape=(total, N_FEAT))
        rec = np.dtype([("ids", "<i4", (TOPK,)), ("lg", "<f2", (TOPK,))])
        self.topk = np.memmap(prefix + ".topk", dtype=rec, mode="r", shape=(total,))
        self.meta = [json.loads(l) for l in open(prefix + ".seqs.bin.meta.jsonl")][: len(self.n)]
        # tokens are re-read from the seqs file to stay aligned with what was dumped
        self.tokens = []
        with open(prefix + ".seqs.bin", "rb") as f:
            for _ in range(len(self.n)):
                (m,) = struct.unpack("<I", f.read(4))
                self.tokens.append(np.frombuffer(f.read(4 * m), dtype=np.uint32).astype(np.int64))
        for i, m in enumerate(self.meta):
            assert m["n"] == self.n[i] == len(self.tokens[i]), (i, m, self.n[i])

    def __len__(self):
        return len(self.n)

    def get(self, i):
        o, n = self.off[i], self.n[i]
        feat = torch.from_numpy(np.ascontiguousarray(self.feat[o:o + n]))
        tk = self.topk[o:o + n]
        topk_ids = torch.from_numpy(np.ascontiguousarray(tk["ids"])).long()
        topk_lg = torch.from_numpy(np.ascontiguousarray(tk["lg"])).float()
        toks = torch.from_numpy(self.tokens[i])
        loss_mask = torch.zeros(n, dtype=torch.bool)
        loss_mask[self.meta[i]["n_prompt"]:] = True
        return {"tokens": toks, "feat": feat, "topk_ids": topk_ids, "topk_lg": topk_lg, "loss_mask": loss_mask, "meta": self.meta[i]}
