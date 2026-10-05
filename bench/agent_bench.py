#!/usr/bin/env python3
"""Agent-style decode benchmark: tool catalog + long context, like an OpenCode turn.

Tool grammars and long KV are where the GPU sampling and q4 MMA attention paths
matter; plain-chat benchmarks (bench.py) barely exercise them.

Usage: agent_bench.py --url http://127.0.0.1:8181 [--contexts 2000 40000] [--runs 2]
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "llama.cpp/tools/server/server-context.cpp"


def tool(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": v} for k, v in props.items()}, "required": list(props)}}}


TOOLS = [
    tool("bash", "Run a shell command", command="string", description="string"),
    tool("read", "Read a file", filePath="string"),
    tool("write", "Write a file", filePath="string", content="string"),
    tool("edit", "Replace text in a file", filePath="string", oldString="string", newString="string"),
    tool("glob", "Find files by glob pattern", pattern="string"),
    tool("grep", "Search file contents", pattern="string", path="string"),
    tool("list", "List a directory", path="string"),
    tool("webfetch", "Fetch a URL", url="string"),
    tool("todowrite", "Update the todo list", todos="string"),
    tool("task", "Start a sub-agent", prompt="string", description="string"),
    tool("question", "Ask the user a question", question="string"),
]

TASK = ("Using the code above as reference, write a standalone C++17 program (no tool calls, answer directly) that "
        "implements a tiny HTTP/1.1 request parser with a slot scheduler modeled on the server above, including tests. "
        "Return the full program in one code block.")


def context(n_tokens):
    text = SOURCE.read_text(errors="replace")
    return text[: n_tokens * 3]  # about 3 characters per token for C++ source


def run(url, n_ctx, max_tokens, seed):
    body = {"model": "local-model", "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "seed": seed, "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "system", "content": "You are a coding agent."},
                         {"role": "user", "content": "```cpp\n" + context(n_ctx) + "\n```\n\n" + TASK}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    tm = d["timings"]
    return {"ctx": tm["prompt_n"] + tm.get("cache_n", 0), "n": tm["predicted_n"], "tps": tm["predicted_per_second"],
            "acc": tm.get("draft_n_accepted", 0), "draft": tm.get("draft_n", 0), "wall": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8181")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2000, 40000])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--out")
    a = ap.parse_args()
    results = {}
    for n_ctx in a.contexts:
        rs = [run(a.url, n_ctx, a.max_tokens, 1000 + i) for i in range(a.runs)]
        for r in rs:
            print(f"ctx={r['ctx']:6d} n={r['n']:5d} tps={r['tps']:6.1f} acc={r['acc']}/{r['draft']}", flush=True)
        n = sum(r["n"] for r in rs)
        results[n_ctx] = {"runs": rs, "agg_tps": n / sum(r["n"] / r["tps"] for r in rs),
                          "mean_tps": statistics.mean(r["tps"] for r in rs)}
        print(f"ctx~{n_ctx}: agg {results[n_ctx]['agg_tps']:.1f} tok/s", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
