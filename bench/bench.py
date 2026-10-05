#!/usr/bin/env python3
"""Single-stream decode benchmark against a llama-server launched per config.

Usage:
  bench.py --name ar            --spec none
  bench.py --name dflash2-q8-n7 --spec dflash --draft models/dflash2-gguf/...Q8_0.gguf --n-max 7
  bench.py --name mtp-n2        --spec mtp --n-max 2

Writes runs/<name>.json with per-prompt timings and a summary.
"""

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import all_prompts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "llama.cpp/build/bin/llama-server"
PORT = 8181


def server_cmd(a):
    cmd = [
        str(a.server_bin), "-m", str(a.model),
        "--host", "127.0.0.1", "--port", str(PORT),
        "-c", str(a.ctx), "-np", "1", "-ngl", "999", "-fa", "on",
        "-ctk", a.kv, "-ctv", a.kv, "-b", "2048", "-ub", "512",
        "--jinja", "--chat-template-file", str(ROOT / "models/chat-template.jinja"),
        "--reasoning-format", "deepseek", "--no-webui", "--poll", "10",
        "-t", "12", "-tb", "16",
    ]
    if a.no_mmap:
        cmd.append("--no-mmap")
    if a.spec == "dflash":
        cmd += ["--spec-type", "draft-dflash", "-md", a.draft, "--spec-draft-ngl", "999",
                "--spec-draft-n-max", str(a.n_max), "--spec-draft-p-min", str(a.p_min)]
    elif a.spec == "mtp":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(a.n_max)]
    elif a.spec == "fastmtp":
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-model", str(ROOT / "models/fastmtp.gguf"),
                "--spec-draft-ngl", "999", "--spec-draft-n-max", str(a.n_max), "--spec-draft-p-min", str(a.p_min),
                "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16"]
    cmd += a.extra
    return cmd


def wait_ready(proc, timeout=600, log_path=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        if log_path is not None:
            text = log_path.read_text(errors="replace")
            if "CUDA driver is a stub" in text or "no usable GPU found" in text:
                raise RuntimeError("CUDA unavailable; refusing a CPU benchmark")
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError("server not ready")


def run_prompt(prompt, a):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": a.max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": a.thinking},
    }
    if a.sampling == "greedy":
        body.update(temperature=0.0, top_k=1)
    else:
        body.update(temperature=1.0, top_p=0.95, top_k=20, seed=a.seed)
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=a.request_timeout)
    r.raise_for_status()
    d = r.json()
    tm = d.get("timings", {})
    msg = d["choices"][0]["message"]
    return {
        "wall_s": time.time() - t0,
        "prompt_n": tm.get("prompt_n"),
        "prompt_tps": tm.get("prompt_per_second"),
        "n": tm.get("predicted_n"),
        "tps": tm.get("predicted_per_second"),
        "draft_n": tm.get("draft_n", 0),
        "draft_acc": tm.get("draft_n_accepted", 0),
        "text": (msg.get("reasoning_content") or "") + "\n<<answer>>\n" + (msg.get("content") or ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--spec", choices=["none", "dflash", "mtp", "fastmtp"], default="none")
    ap.add_argument("--draft")
    ap.add_argument("--n-max", type=int, default=7)
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kv", default="q4_0")
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--sampling", choices=["greedy", "sampled"], default="sampled")
    ap.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--server-bin", default=str(BIN))
    ap.add_argument("--model", default=str(ROOT / "models/target-iq4xs.gguf"))
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--request-timeout", type=float, default=300)
    ap.add_argument("extra", nargs="*", help="extra server args after --")
    a = ap.parse_args()
    if a.spec == "dflash" and not a.draft:
        ap.error("--spec dflash requires --draft")

    prompts = all_prompts(include_long=not a.no_long)
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only}

    (ROOT / "runs").mkdir(exist_ok=True)
    log = open(ROOT / "runs" / f"{a.name}.server.log", "w")
    cmd = server_cmd(a)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(ROOT / "driver-libs") + ":" + env.get("LD_LIBRARY_PATH", "")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    results = {}
    try:
        wait_ready(proc, log_path=ROOT / "runs" / f"{a.name}.server.log")
        run_prompt("Say hi.", a)  # warmup
        for k, p in prompts.items():
            res = run_prompt(p, a)
            if not a.keep_text:
                res["text"] = res["text"][:400]
            results[k] = res
            al = (res["n"] / max(1, res["n"] - res["draft_acc"])) if res["draft_n"] else 1.0
            print(f"{k:18s} ctx={res['prompt_n']:6d} n={res['n']:5d} tps={res['tps']:7.2f} "
                  f"acc={res['draft_acc']}/{res['draft_n']} tok/step={al:.2f}", flush=True)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)
        log.close()

    def summarize(keys):
        rs = [results[k] for k in keys if k in results]
        if not rs:
            return {}
        n = sum(r["n"] for r in rs)
        t = sum(r["n"] / r["tps"] for r in rs)
        steps = sum(r["n"] - r["draft_acc"] for r in rs)
        return {
            "prompts": len(rs),
            "tokens": n,
            "agg_tps": n / t,
            "mean_tps": statistics.mean(r["tps"] for r in rs),
            "tok_per_step": n / max(1, steps),
            "ms_per_step": 1000 * t / max(1, steps),
            "accept_rate": sum(r["draft_acc"] for r in rs) / max(1, sum(r["draft_n"] for r in rs)),
        }

    short = [k for k in results if not k.startswith("long_")]
    longk = [k for k in results if k.startswith("long_")]
    summary = {"all": summarize(list(results)), "short": summarize(short), "long": summarize(longk)}
    overrides = {k: v for k, v in env.items() if k.startswith("GGML_CUDA_") or k == "NVIDIA_TF32_OVERRIDE"}
    out = {"name": a.name, "cmd": cmd, "args": vars(a), "env": overrides, "results": results, "summary": summary}
    (ROOT / "runs" / f"{a.name}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
