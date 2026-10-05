#!/usr/bin/env python3
"""Replay the R20 cohort tasks (balanced-v1 sampling) directly against a server.

Same task prompts and sampling as the OpenCode R20 run, without the OpenCode client and
tool execution. Measures decode tok/s, tokens per verify step and ms per step.

Usage: r20_replay.py [--url http://127.0.0.1:8080] [--tasks story explanation python100]
"""
import argparse
import hashlib
import json
import time
import urllib.request

import agent_bench
from opencode_client_bench import TASKS

PROSE_SYSTEM = ("Write only the complete prose answer requested by the user. Do not use tools, count words "
                "with code, create files, ask questions, or delegate.")


def body(name):
    prose = name in ("story", "explanation")
    b = {"model": "local-model", "max_tokens": 20000, "temperature": 0.4 if prose else 0.0, "top_p": 0.95,
         "top_k": 20, "min_p": 0, "presence_penalty": 0, "seed": 1234,
         "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
         "messages": [{"role": "system", "content": PROSE_SYSTEM if prose else "You are a coding agent."},
                      {"role": "user", "content": TASKS[name]}]}
    if not prose:
        b["tools"] = agent_bench.TOOLS
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tasks", nargs="+", default=["story", "explanation", "python100"])
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for name in a.tasks:
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body(name)).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        tm, msg = d["timings"], d["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + "\0" + (msg.get("content") or "") + "\0" + json.dumps(msg.get("tool_calls"))
        n, acc, tps = tm["predicted_n"], tm.get("draft_n_accepted", 0), tm["predicted_per_second"]
        steps = max(1, n - acc)
        row = {"task": name, "n": n, "tps": tps, "accept": acc / max(1, tm.get("draft_n", 1)),
               "tok_per_step": n / steps, "ms_per_step": 1000 * n / tps / steps, "wall": time.time() - t0,
               "sha": hashlib.sha256(text.encode()).hexdigest()[:16]}
        rows.append(row)
        print(f"{name:12s} n={n:6d} tps={tps:6.1f} accept={row['accept']:.3f} tok/step={row['tok_per_step']:.2f} "
              f"ms/step={row['ms_per_step']:.1f} sha={row['sha']}", flush=True)
    n = sum(r["n"] for r in rows)
    print(f"weighted {n / sum(r['n'] / r['tps'] for r in rows):.1f} tok/s over {n} tokens")
    if a.out:
        open(a.out, "w").write(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
