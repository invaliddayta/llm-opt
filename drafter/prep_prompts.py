#!/usr/bin/env python3
"""Build a mixed prompt pool for self-distillation of the DFlash2 drafter.

Output: data/prompts.jsonl, one {"id", "source", "messages"} per line. The
messages end with a user turn; the target model writes the assistant turn.
"""

import json
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "prompts.jsonl"
rng = random.Random(0)


def dl(repo, f):
    return hf_hub_download(repo, f, repo_type="dataset")


def ok_len(s, lo=16, hi=6000):
    return isinstance(s, str) and lo <= len(s) <= hi


def wildchat(n):
    out = []
    for shard in range(2):
        df = pd.read_parquet(dl("allenai/WildChat-1M", f"data/train-0000{shard}-of-00014.parquet"),
                             columns=["conversation", "language", "toxic", "redacted"])
        for conv, lang, toxic in zip(df.conversation, df.language, df.toxic):
            if toxic or len(conv) == 0:
                continue
            u = conv[0]["content"]
            if ok_len(u):
                out.append({"source": f"wildchat-{lang}", "messages": [{"role": "user", "content": u}]})
    rng.shuffle(out)
    # keep mostly English but some multilingual
    en = [x for x in out if x["source"] == "wildchat-English"]
    other = [x for x in out if x["source"] != "wildchat-English"]
    k_other = n // 5
    return en[: n - k_other] + other[:k_other]


def ultrachat(n):
    df = pd.read_parquet(dl("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"))
    out = []
    for msgs in df.messages:
        msgs = list(msgs)
        # multi-turn: keep a random prefix ending at a user turn
        user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        cut = rng.choice(user_idx[:3])
        conv = [{"role": m["role"], "content": m["content"]} for m in msgs[: cut + 1]]
        if all(ok_len(m["content"], 1, 8000) for m in conv):
            out.append({"source": "ultrachat", "messages": conv})
    rng.shuffle(out)
    return out[:n]


def magicoder(n):
    p = dl("ise-uiuc/Magicoder-Evol-Instruct-110K", "data-evol_instruct-decontaminated.jsonl")
    out = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if ok_len(d["instruction"]):
                out.append({"source": "magicoder", "messages": [{"role": "user", "content": d["instruction"]}]})
    rng.shuffle(out)
    return out[:n]


def opencode(n):
    df = pd.read_parquet(dl("nvidia/OpenCodeInstruct", "data/train-00000-of-00050.parquet"), columns=["input"])
    out = [{"source": "opencode", "messages": [{"role": "user", "content": s}]} for s in df.input if ok_len(s)]
    rng.shuffle(out)
    return out[:n]


def math(n):
    df = pd.read_parquet(dl("AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet"), columns=["problem", "source"])
    out = [{"source": "numina", "messages": [{"role": "user", "content": s}]} for s in df.problem if ok_len(s)]
    rng.shuffle(out)
    g = pd.read_parquet(dl("openai/gsm8k", "main/train-00000-of-00001.parquet"))
    gs = [{"source": "gsm8k", "messages": [{"role": "user", "content": s}]} for s in g.question]
    rng.shuffle(gs)
    return out[: n - n // 3] + gs[: n // 3]


def hermes_fc(n):
    out = []
    for f in ["func-calling.json", "json-mode-agentic.json", "func-calling-singleturn.json"]:
        data = json.load(open(dl("NousResearch/hermes-function-calling-v1", f)))
        for d in data:
            conv = []
            for m in d["conversations"]:
                role = {"system": "system", "human": "user", "gpt": "assistant", "tool": "user"}[m["from"]]
                content = m["value"]
                if m["from"] == "tool":
                    content = "Tool result:\n" + content
                conv.append({"role": role, "content": content})
            # cut at a random user turn (after the system prompt)
            user_idx = [i for i, m in enumerate(conv) if m["role"] == "user"]
            if not user_idx:
                continue
            cut = rng.choice(user_idx)
            conv = conv[: cut + 1]
            # merge consecutive same-role messages
            merged = []
            for m in conv:
                if merged and merged[-1]["role"] == m["role"]:
                    merged[-1]["content"] += "\n\n" + m["content"]
                else:
                    merged.append(dict(m))
            if sum(len(m["content"]) for m in merged) < 20000:
                out.append({"source": f"hermesfc-{f.split('.')[0]}", "messages": merged})
    rng.shuffle(out)
    return out[:n]


def main(total=12000):
    mix = {
        wildchat: 0.30,
        ultrachat: 0.12,
        magicoder: 0.12,
        opencode: 0.08,
        math: 0.16,
        hermes_fc: 0.22,
    }
    rows = []
    for fn, frac in mix.items():
        got = fn(int(total * frac))
        print(fn.__name__, len(got))
        rows += got
    rng.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i, r in enumerate(rows):
            r["id"] = i
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), OUT)


if __name__ == "__main__":
    main()
