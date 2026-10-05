#!/usr/bin/env python3
"""Benchmark real OpenCode runs through a transparent loopback capture proxy.

Only the isolated workspace config is written. Model requests/responses and CLI
events remain local; credentials are never recorded. No request fields are
changed by the proxy. An optional direct replay uses exactly the captured body.
"""

import argparse
import ast
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import zlib
import cohort_profile


TASKS = {
    "python100": "Write a useful custom Python 3 program of about 100 lines. Choose the task yourself, but use only the standard library, include a CLI, validation, and error handling. Return the complete program in a single code block. Do not use tools or create files; I want the generated code in your answer.",
    "python_long": "Choose a genuinely useful Python 3 application and write its complete implementation, around 500 lines, with standard-library-only dependencies, a CLI, careful error handling, and internal tests. No placeholder functions or repeated padding. Return the complete code in your answer without using tools or creating files.",
    "explanation": "Explain how a database uses write-ahead logging, transactions, and checkpoints to recover after a power failure. Write roughly 1200 words of clear continuous prose for an interested beginner, using a worked example and discussing tradeoffs. Do not use tools or create files.",
    "story": "Write an original short story of roughly 1500 words about a bicycle mechanic who discovers that a customer's map describes a town that no longer exists. Use natural dialogue, distinct characters, and a satisfying ending. Do not use tools or create files.",
    "python_tools": "In this workspace, create a useful custom Python 3 command-line program of about 100 lines using only the standard library. Choose its purpose yourself. Also write meaningful unittest tests and run them, fixing any failures. Do not access any paths outside this workspace, use network tools, or delegate. Finish with a short summary of the program and test results.",
    "python_tools_long": "In this workspace, implement a complete standard-library-only Python 3 SQLite personal-finance CLI in budget.py, roughly 450-550 lines of real implementation, plus meaningful unittest tests in test_budget.py. Include account creation/listing, transaction add/list, atomic transfers, CSV import/export, monthly reports and balances, exact cent-based money handling, validation, schema setup, and useful command-line errors. Do not use placeholder functions or padding. Make one brief plan, then create the files, run the tests, fix failures, and stop when they pass. Do not access paths outside this workspace, use network tools, or delegate. Finish with a short feature and test summary.",
}


def stop_owned(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


class Capture:
    def __init__(self, root):
        self.root = root
        self.records = []
        self.lock = threading.Lock()
        self.tag = "startup"

    def begin(self, body, path):
        with self.lock:
            index = len(self.records)
            record = {"index": index, "tag": self.tag, "path": path,
                      "started_ns": time.perf_counter_ns(), "started_wall": time.time(),
                      "request": json.loads(body), "events": [], "bytes": 0}
            self.records.append(record)
            return record

    def save(self, record):
        serial = {k: v for k, v in record.items() if k != "events" and not k.startswith("_")}
        (self.root / f"wire-{record['index']:03d}.json").write_text(json.dumps(serial, indent=2))


def consume_sse(record, buffer, chunk, final=False):
    buffer += chunk
    if final and buffer.endswith(b"\r"):
        buffer += b"\n"
    data_lines = record.setdefault("_data_lines", [])
    while match := re.search(rb"\r\n|\r(?!$)|\n", buffer):
        line, buffer = buffer[:match.start()], buffer[match.end():]
        if line:
            if line.startswith(b"data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(b" ") else value)
            continue
        if not data_lines:
            continue
        data = b"\n".join(data_lines)
        data_lines.clear()
        now = time.perf_counter_ns()
        if data == b"[DONE]":
            record["done_ns"] = now
            continue
        obj = json.loads(data)
        record["events"].append(obj)
        if obj.get("timings"):
            record["timings"] = obj["timings"]
        if obj.get("usage"):
            record["usage"] = obj["usage"]
        if obj.get("error"):
            record["upstream_error"] = obj["error"]
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            for field in ("content", "reasoning_content"):
                if delta.get(field):
                    record.setdefault(f"first_{field}_ns", now)
                    record[f"{field}_characters"] = record.get(f"{field}_characters", 0) + len(delta[field])
            if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")):
                record.setdefault("first_output_ns", now)
                record["last_output_ns"] = now
            if choice.get("finish_reason"):
                record["finish_reason"] = choice["finish_reason"]
    return buffer


def response_decoder(response):
    encoding = response.getheader("Content-Encoding", "identity").lower()
    if encoding in {"identity", ""}:
        return lambda data: data
    if encoding in {"gzip", "deflate"}:
        decoder = zlib.decompressobj(31 if encoding == "gzip" else 15)
        return decoder.decompress
    raise RuntimeError(f"Cannot capture encoded response: {encoding}")


def handler_factory(capture, target_port):
    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def forward(self):
            record = None
            conn = http.client.HTTPConnection("127.0.0.1", target_port, timeout=600)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 32 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = self.rfile.read(size)
                if self.command == "POST" and self.path.endswith("/chat/completions"):
                    record = capture.begin(body, self.path)
                headers = {k: v for k, v in self.headers.items()
                           if k.lower() not in {"host", "connection", "transfer-encoding"}}
                headers["Connection"] = "close"
                conn.request(self.command, self.path, body=body, headers=headers)
                response = conn.getresponse()
                self.send_response(response.status)
                hop = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
                       "transfer-encoding", "upgrade", "content-length", "server", "date"}
                hop.update(x.strip().lower() for x in response.getheader("Connection", "").split(","))
                for k, v in response.getheaders():
                    if k.lower() not in hop:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                sse = "text/event-stream" in response.getheader("Content-Type", "")
                if record is not None:
                    record["status"] = response.status
                    record["headers_ns"] = time.perf_counter_ns()
                remainder = b""
                raw = bytearray()
                decode = response_decoder(response)
                decoded = bytearray()
                while chunk := response.read1(65536):
                    if record is not None:
                        record["bytes"] += len(chunk)
                        raw.extend(chunk)
                        plain = decode(chunk)
                        decoded.extend(plain)
                        if sse:
                            remainder = consume_sse(record, remainder, plain)
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if record is not None:
                    if sse:
                        consume_sse(record, remainder, b"", final=True)
                    record["response_ns"] = time.perf_counter_ns()
                    (capture.root / f"wire-{record['index']:03d}.response").write_bytes(raw)
                    if not sse:
                        obj = json.loads(decoded)
                        record["timings"] = obj.get("timings", {})
                        record["usage"] = obj.get("usage", {})
                        record["nonstream_complete"] = True
                    capture.save(record)
            except Exception as exc:
                if record is not None:
                    record["error"] = repr(exc)
                    capture.save(record)
                self.close_connection = True
            finally:
                conn.close()

    return Proxy


def validate_record(record):
    if record.get("error") or record.get("upstream_error") or record.get("status") != 200:
        raise ValueError(f"Failed HTTP turn {record['index']}")
    if "response_ns" not in record or not (record.get("done_ns") or record.get("nonstream_complete")):
        raise ValueError(f"Incomplete HTTP turn {record['index']}")
    tm = record.get("timings", {})
    for k in ["predicted_n", "predicted_ms", "predicted_per_second", "prompt_n", "prompt_ms"]:
        if not isinstance(tm.get(k), (int, float)) or not math.isfinite(tm[k]) or tm[k] < 0:
            raise ValueError(f"Missing/invalid {k} in HTTP turn {record['index']}")
    if tm["predicted_n"] < 1 or tm["predicted_ms"] <= 0:
        raise ValueError("No measurable completed decode")
    calculated = 1000 * (tm["predicted_n"] - 1) / tm["predicted_ms"]
    if not math.isclose(tm["predicted_per_second"], calculated, rel_tol=1e-4, abs_tol=0.01):
        raise ValueError("Server rate/token convention mismatch")
    if not record.get("nonstream_complete") and "first_output_ns" not in record:
        raise ValueError("Completed output lacks stream timestamps")


def summarize(records, wall_s):
    if not records:
        raise ValueError("No captured model turns")
    turns = []
    for r in records:
        validate_record(r)
        tm = r.get("timings", {})
        request = r["request"]
        turns.append({
            "wire_index": r["index"], "status": r.get("status"),
            "finish_reason": r.get("finish_reason"), "error": r.get("error") or r.get("upstream_error"),
            "sampling": {k: v for k, v in request.items() if k not in {"messages", "tools"}},
            "tools": len(request.get("tools", [])), "messages": len(request.get("messages", [])),
            "timings": tm, "usage": r.get("usage"),
            "ttft_s": (r["first_output_ns"] - r["started_ns"]) / 1e9 if "first_output_ns" in r else None,
            "time_to_visible_answer_s": (r["first_content_ns"] - r["started_ns"]) / 1e9 if "first_content_ns" in r else None,
            "answer_characters": r.get("content_characters", 0),
            "reasoning_characters": r.get("reasoning_content_characters", 0),
            "http_wall_s": (r["response_ns"] - r["started_ns"]) / 1e9,
        })
    n = sum(t["timings"].get("predicted_n", 0) for t in turns)
    decode_s = sum(t["timings"].get("predicted_ms", 0) / 1000 for t in turns)
    prompt_s = sum(t["timings"].get("prompt_ms", 0) / 1000 for t in turns)
    decoded_n = sum(t["timings"]["predicted_per_second"] * t["timings"]["predicted_ms"] / 1000 for t in turns)
    return {"turns": turns, "wall_s": wall_s, "output_tokens": n, "decode_tokens": decoded_n, "decode_s": decode_s,
            "prompt_s": prompt_s, "aggregate_decode_tps": decoded_n / decode_s,
            "end_to_end_tps": n / wall_s if wall_s else None,
            "outside_model_s": wall_s - decode_s - prompt_s}


def direct_replay(record, port, root, timeout):
    body = json.dumps(record["request"]).encode()
    replay = {"index": record["index"], "request": record["request"],
              "started_ns": time.perf_counter_ns(), "events": []}
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=min(timeout, 30))
    expired = threading.Event()
    active_socket = []
    deadline = time.monotonic() + timeout
    def expire():
        expired.set()
        sock = active_socket[0] if active_socket else conn.sock
        if sock:
            try:
                sock.shutdown(2)
            except OSError:
                pass
        conn.close()
    watchdog = threading.Timer(timeout, expire)
    watchdog.start()
    try:
        conn.request("POST", record["path"], body, {"Content-Type": "application/json"})
        active_socket.append(conn.sock)
        if expired.is_set():
            expire()
            raise TimeoutError("Direct replay overall deadline expired")
        response = conn.getresponse()
        replay["status"] = response.status
        remainder = b""
        raw = bytearray()
        decode = response_decoder(response)
        decoded = bytearray()
        sse = "text/event-stream" in response.getheader("Content-Type", "")
        while chunk := response.read1(65536):
            if time.monotonic() >= deadline:
                raise TimeoutError("Direct replay overall deadline expired")
            raw.extend(chunk)
            plain = decode(chunk)
            decoded.extend(plain)
            if sse:
                remainder = consume_sse(replay, remainder, plain)
        if expired.is_set():
            raise TimeoutError("Direct replay overall deadline expired")
        if sse:
            consume_sse(replay, remainder, b"", final=True)
        else:
            obj = json.loads(decoded)
            replay["timings"] = obj.get("timings", {})
            replay["usage"] = obj.get("usage", {})
            replay["nonstream_complete"] = True
        replay["response_ns"] = time.perf_counter_ns()
        (root / f"replay-{record['index']:03d}.response").write_bytes(raw)
    finally:
        watchdog.cancel()
        conn.close()
    return summarize([replay], (replay["response_ns"] - replay["started_ns"]) / 1e9)


def api(a, method, path, body=None):
    cmd = ["opencode", "api", method, path]
    if body is not None:
        cmd += ["--data", json.dumps(body)]
    if a.server:
        cmd += ["--server", a.server]
    result = subprocess.run(cmd, cwd=a.workspace, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout) if result.stdout.strip() else None


def cli_tool_metrics(path):
    tools = set()
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("type") != "tool_use":
            continue
        tools.add(event["part"]["id"])
    # CLI start/end includes streamed tool input, not just execution.
    return {"tool_calls": len(tools), "tool_time_valid": not tools, "tool_execution_s": None if tools else 0.0}


def tool_metrics(export):
    tools = {}
    def visit(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "tool" and "state" in obj:
                tools[obj["id"]] = obj
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    visit(export)
    intervals = []
    for part in tools.values():
        tm = part.get("time", {})
        if "ran" not in tm or "completed" not in tm:
            return {"tool_calls": len(tools), "tool_time_valid": False, "tool_execution_s": None}
        intervals.append((tm["ran"], tm["completed"]))
    intervals.sort()
    union = []
    for beg, end in intervals:
        if end < beg:
            raise ValueError("Invalid tool interval")
        if union and beg <= union[-1][1]:
            union[-1][1] = max(end, union[-1][1])
        else:
            union.append([beg, end])
    return {"tool_calls": len(tools), "tool_time_valid": True,
            "tool_execution_s": sum(end - beg for beg, end in union) / 1000}


def python_snapshot(workspace):
    result = {}
    for path in workspace.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if not path.resolve().is_relative_to(workspace.resolve()):
            raise ValueError(f"Python artifact escapes workspace: {path}")
        result[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def validate_delivery(name, summary, records, workspace, before):
    """Minimum delivery gate, not a substitute for reviewing generated functionality."""
    answer = "".join(choice.get("delta", {}).get("content") or "" for record in records
                     for event in record.get("events", []) for choice in event.get("choices", []))
    after = python_snapshot(workspace)
    changed = sorted(path for path, digest in after.items() if before.get(path) != digest)
    failures = []
    tools = summary.get("tool_calls", 0)
    if not answer.strip():
        failures.append("No visible answer or completion summary")
    if name in {"python_tools", "python_tools_long"}:
        if not tools:
            failures.append("Required tool workflow did not occur")
        expected = {"budget.py", "test_budget.py"} if name == "python_tools_long" else set()
        if expected - set(changed):
            failures.append("Required files missing or unchanged: " + ", ".join(sorted(expected - set(changed))))
        programs, tests = [], []
        for path in changed:
            try:
                tree = ast.parse((workspace / path).read_text())
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"Invalid Python artifact {path}: {exc}")
                continue
            functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            test_count = sum(node.name.startswith("test_") for node in functions)
            if Path(path).name.startswith("test") and test_count:
                tests.append(path)
            elif functions:
                programs.append(path)
            for node in functions:
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                    body = body[1:]
                if not body or all(isinstance(stmt, ast.Pass) for stmt in body):
                    failures.append(f"Stub function: {path}:{node.name}")
        if not programs or not tests:
            failures.append("No changed implementation plus meaningful test module")
    else:
        tests = []
        if tools or changed:
            failures.append("Answer-only task used tools or changed Python files")
        if name in {"python100", "python_long"}:
            blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, flags=re.DOTALL)
            try:
                if len(blocks) != 1:
                    raise ValueError("Expected one complete Python code block")
                ast.parse(blocks[0])
            except (SyntaxError, ValueError) as exc:
                failures.append(str(exc))
    return {"valid": not failures, "failures": failures, "changed_python_files": changed,
            "before_sha256": before, "after_sha256": after, "test_modules": tests,
            "scope": "minimum delivery only; feature quality requires independent review"}


def verify_generated_tests(workspace, tests, log_path):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="verify-pycache-", dir=os.environ.get("OPENCODE_BENCH_VERIFY_TMPDIR", "/tmp/opencode")) as cache, log_path.open("w") as log:
        cmd = [sys.executable, "-X", f"pycache_prefix={cache}", "-m", "unittest", "-v", *tests]
        proc = subprocess.Popen(cmd, cwd=workspace, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop_owned(proc)
    text = log_path.read_text(errors="replace")
    counts = re.findall(r"^Ran (\d+) tests? in ", text, re.MULTILINE)
    count = int(counts[-1]) if counts else 0
    return {"valid": not timed_out and proc.returncode == 0 and count > 0,
            "returncode": proc.returncode, "timed_out": timed_out, "test_count": count,
            "wall_s": time.perf_counter() - start, "log": str(log_path), "command": cmd,
            "timing_note": "Independent verification after task, excluded from client wall/decode times"}


def check_task_delivery(name, summary, records, workspace, before, log_path):
    try:
        summary["delivery"] = validate_delivery(name, summary, records, workspace, before)
        if summary["delivery"]["valid"] and summary["delivery"]["test_modules"]:
            summary["independent_tests"] = verify_generated_tests(workspace, summary["delivery"]["test_modules"], log_path)
        if not summary["delivery"]["valid"] or not summary.get("independent_tests", {"valid": True})["valid"]:
            summary["failed"] = True
            summary["reason"] = "Task delivery or independent generated tests failed"
    except Exception as exc:
        summary["failed"] = True
        summary["reason"] = f"Task delivery validation exception: {exc!r}"
        summary["delivery"] = {"valid": False, "failures": [summary["reason"]]}


def sampling_body(temperature, thinking):
    return {"temperature": temperature, "top_p": 0.95, "top_k": 20, "min_p": 0,
            "presence_penalty": 0, "seed": 1234,
            "chat_template_kwargs": {"enable_thinking": thinking == "on", "preserve_thinking": True}}


def task_preset(name, profile="original"):
    if profile != "original":
        return cohort_profile.preset(name, profile, task_preset, sampling_body)
    if name == "story":
        return "gpu-story", sampling_body(1.0, "on") | {"reasoning_effort": "xhigh"}
    if name == "explanation":
        return "gpu-technical", sampling_body(0.4, "on") | {"reasoning_effort": "xhigh"}
    return "gpu-code", sampling_body(0.0, "off") | {"reasoning_effort": "xhigh"}


def sampling_matches(records, expected_body):
    return bool(records) and all(all(record["request"].get(key) == value for key, value in expected_body.items()) for record in records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument("--proxy-port", type=int, default=8190)
    ap.add_argument("--target-port", type=int, default=8080)
    ap.add_argument("--server", help="Optional explicit OpenCode server URL, otherwise use its shared service")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--explicit-sampling", action="store_true", help="Send documented body overrides, not legacy model options")
    ap.add_argument("--temperature", type=float, default=1.0, help="Isolated sampler experiment; existing default unchanged")
    ap.add_argument("--thinking", choices=("on", "off"), default="on", help="Isolated reasoning-mode experiment; existing default unchanged")
    ap.add_argument("--task-presets", action="store_true", help="Isolated per-task variant experiment; not unchanged-mode optimization evidence")
    ap.add_argument("--cohort-profile", choices=cohort_profile.PROFILES, default="original", help="Explicit new sampler/agent cohort; original TASKS and previous presets stay unchanged")
    a = ap.parse_args()
    if not math.isfinite(a.temperature) or not 0 <= a.temperature <= 2 or (a.temperature != 1 and not a.explicit_sampling):
        ap.error("temperature requires explicit sampling and a finite value in 0..2")
    if a.thinking != "on" and not a.explicit_sampling:
        ap.error("thinking override requires explicit sampling")
    if a.task_presets and (not a.explicit_sampling or a.temperature != 1 or a.thinking != "on"):
        ap.error("task presets require explicit sampling and unchanged base temperature/thinking options")
    if a.cohort_profile != "original" and (not a.task_presets or any(name not in ("story", "explanation", "python_tools_long", "python_tools") for name in a.tasks)):
        ap.error("new cohort profiles require explicit representative task presets")
    a.workspace = a.workspace.resolve()
    a.output = a.output.resolve()
    if a.server:
        url = urlsplit(a.server)
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Credential-bearing server URLs are not accepted; use OpenCode's authentication context")
    a.output.mkdir(parents=True, exist_ok=False)
    a.workspace.mkdir(parents=True, exist_ok=True)
    config_path = a.workspace / "opencode.jsonc"
    if config_path.exists():
        raise RuntimeError(f"Refusing to overwrite {config_path}")
    model = {}
    if a.explicit_sampling:
        model["body"] = sampling_body(a.temperature, a.thinking)
    if a.task_presets:
        presets = dict(task_preset(name, a.cohort_profile) for name in a.tasks)
        model["variants"] = [{"id": variant, "body": body} for variant, body in presets.items()]
    config = {"$schema": "https://opencode.ai/config.json", "model": "qwen-local/local-model",
              "small_model": "qwen-local/local-model", "providers": {"qwen-local": {
                  "settings": {"baseURL": f"http://127.0.0.1:{a.proxy_port}/v1"},
                  "models": {"local-model": model}}},
              "agents": {"build": {"model": "qwen-local/local-model#xhigh",
                                     "permissions": [{"action": "subagent", "resource": "*", "effect": "deny"}]}}}
    config["agents"].update(cohort_profile.agents(a.cohort_profile))
    config_path.write_text(json.dumps(config, indent=2))
    capture = Capture(a.output)
    proxy = ThreadingHTTPServer(("127.0.0.1", a.proxy_port), handler_factory(capture, a.target_port))
    proxy.daemon_threads = True
    thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    thread.start()
    results = {}
    try:
        for name in a.tasks:
            capture.tag = name
            variant, expected_body = task_preset(name, a.cohort_profile) if a.task_presets else ("xhigh", sampling_body(a.temperature, a.thinking))
            agent = cohort_profile.agent(name, a.cohort_profile)
            experiment = {"model_variant": variant, "expected_sampling_body": expected_body if a.explicit_sampling else None,
                          "cohort_profile": a.cohort_profile, "agent": agent}
            results[name] = experiment | {"failed": True, "reason": "Task did not reach completion", "outcome": None}
            (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
            try:
                before = python_snapshot(a.workspace)
            except Exception as exc:
                results[name] = experiment | {"failed": True, "reason": f"Artifact snapshot exception: {exc!r}", "outcome": None}
                (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
                print(f"{name}: FAILED workspace preflight: {exc!r}", flush=True)
                continue
            setup_start = time.perf_counter()
            created = api(a, "post", "/api/session", {"title": f"Local performance: {name}", "agent": agent,
                "model": {"providerID": "qwen-local", "id": "local-model", "variant": variant},
                "location": {"directory": str(a.workspace)}})["data"]
            session_id = created["id"]
            if created["location"]["directory"] != str(a.workspace):
                raise RuntimeError("OpenCode ignored explicit benchmark location")
            setup_s = time.perf_counter() - setup_start
            cmd = ["opencode", "run", "--model", f"qwen-local/local-model#{variant}", "--agent", agent,
                   "--format", "json", "--session", session_id, TASKS[name]]
            if a.server:
                cmd += ["--server", a.server]
            t0 = time.perf_counter()
            with (a.output / f"{name}.events.jsonl").open("w") as out, (a.output / f"{name}.stderr").open("w") as err:
                proc = subprocess.Popen(cmd, cwd=a.workspace, stdout=out, stderr=err, start_new_session=True)
                timed_out = False
                try:
                    proc.wait(timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    cli_wall_s = time.perf_counter() - t0
                    stop_owned(proc)
                    api(a, "post", f"/api/session/{session_id}/interrupt")
                    active = api(a, "get", "/api/session/active")["data"]
                    if session_id in active:
                        raise RuntimeError(f"Owned session {session_id} remained active after interruption")
            if timed_out:
                info = api(a, "get", f"/api/session/{session_id}")["data"]
                (a.output / f"{name}.session-info.json").write_text(json.dumps(info, indent=2))
                results[name] = experiment | {"failed": True, "reason": "task deadline exceeded", "timeout_s": a.timeout,
                                 "wall_s": cli_wall_s, "returncode": proc.returncode, "session_ids": [session_id], "outcome": info.get("outcome")}
                (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
                print(f"{name}: FAILED task deadline ({a.timeout}s); excluded from completed-task throughput", flush=True)
                continue
            records = [r for r in capture.records if r["tag"] == name]
            try:
                summary = summarize(records, cli_wall_s)
            except ValueError as exc:
                info = api(a, "get", f"/api/session/{session_id}")["data"]
                (a.output / f"{name}.session-info.json").write_text(json.dumps(info, indent=2))
                results[name] = experiment | {"failed": True, "reason": str(exc), "wall_s": cli_wall_s,
                                 "returncode": proc.returncode, "session_ids": [session_id], "outcome": info.get("outcome")}
                (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
                print(f"{name}: FAILED invalid model turn: {exc}", flush=True)
                continue
            summary["setup_s"] = setup_s
            summary.update(experiment)
            summary["returncode"] = proc.returncode
            summary["prompt"] = TASKS[name]
            summary["session_ids"] = [session_id]
            try:
                exported = api(a, "get", f"/api/experimental/session/{session_id}/export")["data"]
                (a.output / f"{name}.session.json").write_text(json.dumps(exported, indent=2))
                summary["outcome"] = exported["info"].get("outcome")
                summary.update(tool_metrics(exported))
            except json.JSONDecodeError:
                info = api(a, "get", f"/api/session/{session_id}")["data"]
                (a.output / f"{name}.session-info.json").write_text(json.dumps(info, indent=2))
                summary["outcome"] = info.get("outcome")
                summary["export_note"] = "CLI export was not valid complete JSON; session info and captured CLI events retained instead"
                summary.update(cli_tool_metrics(a.output / f"{name}.events.jsonl"))
            summary["unattributed_overhead_s"] = summary["outside_model_s"] - summary["tool_execution_s"] if summary["tool_time_valid"] else None
            results[name] = summary
            if not records or proc.returncode or summary["outcome"] != "succeeded" or any(t["error"] or t["status"] != 200 for t in summary["turns"]):
                summary["failed"] = True
                summary["reason"] = "CLI/session/model turn did not complete successfully"
                (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
                print(f"{name}: FAILED outcome={summary['outcome']} returncode={proc.returncode}", flush=True)
                continue
            check_task_delivery(name, summary, records, a.workspace, before, a.output / f"{name}.independent-tests.log")
            if a.explicit_sampling and not sampling_matches(records, expected_body):
                summary["failed"] = True
                summary["reason"] = "Actual client sampler body did not match explicit experiment"
            if summary.get("failed"):
                (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
                print(f"{name}: FAILED {summary['reason']}; delivery={summary['delivery']['failures']}; tests={summary.get('independent_tests')}", flush=True)
                continue
            if a.replay:
                summary["direct_replay"] = [direct_replay(r, a.target_port, a.output, a.timeout) for r in records]
            (a.output / "results.json").write_text(json.dumps({"args": vars(a) | {"workspace": str(a.workspace), "output": str(a.output)}, "results": results}, indent=2))
            print(f"{name}: {summary['output_tokens']} tokens, decode={summary['aggregate_decode_tps']:.2f} tok/s, "
                  f"wall={summary['wall_s']:.2f}s, end-to-end={summary['end_to_end_tps']:.2f} tok/s, turns={len(records)}", flush=True)
    finally:
        proxy.shutdown()
        proxy.server_close()
    completed = [r for r in results.values() if not r.get("failed")]
    if not completed:
        raise RuntimeError("No benchmark tasks completed")
    n = sum(r["decode_tokens"] for r in completed)
    sec = sum(r["decode_s"] for r in completed)
    print(json.dumps({"aggregate_decode_tps": n / sec, "mean_task_decode_tps": statistics.mean(
        r["aggregate_decode_tps"] for r in completed), "decode_tokens": n,
        "output_tokens": sum(r["output_tokens"] for r in completed), "completed_tasks": len(completed),
        "failed_tasks": [name for name, r in results.items() if r.get("failed")]}, indent=2))
    if len(completed) != len(results):
        raise RuntimeError("One or more benchmark tasks failed; see retained results")


if __name__ == "__main__":
    main()
