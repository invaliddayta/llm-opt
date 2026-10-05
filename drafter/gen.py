#!/usr/bin/env python3
"""Self-distillation: let the deployed target model answer the prompt pool.

Starts llama-server (target only, many parallel slots, no speculation) and
writes data/gen/<shard>.jsonl rows: {id, source, effort, prompt_tokens, gen_tokens, stop}.
Exact token ids come from /tokenize (prompt) and return_tokens (generation), so
the drafter sees exactly what the target produced.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PORT = 8182
URL = f"http://127.0.0.1:{PORT}"


def start_server(np_, ctx_per_slot, log):
    cmd = [
        str(ROOT / "llama.cpp/build/bin/llama-server"), "-m", str(ROOT / "models/target-iq4xs.gguf"),
        "--host", "127.0.0.1", "--port", str(PORT), "-np", str(np_), "-c", str(np_ * ctx_per_slot),
        "-ngl", "999", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-b", "2048", "-ub", "512",
        "--jinja", "--chat-template-file", str(ROOT / "models/chat-template.jinja"),
        "--no-webui", "--kv-unified", "--cache-reuse", "0", "-t", "12",
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(ROOT / "driver-libs")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    for _ in range(600):
        if p.poll() is not None:
            raise RuntimeError("server died")
        try:
            if requests.get(URL + "/health", timeout=2).status_code == 200:
                return p
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError


def pick_effort(rng):
    r = rng.random()
    if r < 0.40:
        return "xhigh"
    if r < 0.75:
        return "medium"
    if r < 0.90:
        return "low"
    return "off"


def one(row, a, rng_seed):
    rng = random.Random(rng_seed)
    effort = pick_effort(rng)
    kw = {"enable_thinking": False} if effort == "off" else {"reasoning_effort": effort}
    tmpl = requests.post(URL + "/apply-template", json={"messages": row["messages"], "chat_template_kwargs": kw}, timeout=60).json()
    prompt = tmpl["prompt"]
    ptoks = requests.post(URL + "/tokenize", json={"content": prompt, "add_special": False, "parse_special": True}, timeout=60).json()["tokens"]
    if len(ptoks) > a.max_prompt:
        return None
    body = {
        "prompt": ptoks, "n_predict": a.max_gen, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "seed": rng_seed, "return_tokens": True, "cache_prompt": False,
    }
    r = requests.post(URL + "/completion", json=body, timeout=7200).json()
    return {
        "id": row["id"], "source": row["source"], "effort": effort,
        "prompt_tokens": ptoks, "gen_tokens": r["tokens"],
        "stop": r.get("stop_type") or r.get("stopped_eos"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--np", type=int, default=16)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--max-gen", type=int, default=2048)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / "data/prompts.jsonl")][a.start:a.start + a.count]
    out_dir = ROOT / "data/gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{a.shard}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in open(out_path)}
    rows = [r for r in rows if r["id"] not in done]

    log = open(out_dir / f"{a.shard}.server.log", "w")
    proc = start_server(a.np, a.max_prompt + a.max_gen + 64, log)
    t0 = time.time()
    ntok = 0
    try:
        with open(out_path, "a") as fout, ThreadPoolExecutor(a.np) as ex:
            futs = [ex.submit(one, r, a, 1000003 * r["id"] + 17) for r in rows]
            for i, f in enumerate(futs):
                try:
                    res = f.result()
                except Exception as e:  # keep going on single failures
                    print("error", e, flush=True)
                    continue
                if res is None:
                    continue
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                ntok += len(res["gen_tokens"])
                if i % 20 == 0:
                    dt = time.time() - t0
                    print(f"{i+1}/{len(rows)} gen_tokens={ntok} {ntok/dt:.1f} tok/s", flush=True)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)


if __name__ == "__main__":
    main()
