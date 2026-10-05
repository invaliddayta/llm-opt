#!/usr/bin/env python3
"""CPU-only tests for benchmark framing, accounting, and deadlines."""
import gzip
import http.client
import json
import contextlib
import io
import os
from pathlib import Path
import py_compile
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import opencode_client_bench as bench
import analyze_client_bench as analyze


TIMINGS = {"predicted_n": 2, "predicted_ms": 10, "predicted_per_second": 100,
           "prompt_n": 3, "prompt_ms": 5}
EVENT = {"choices": [{"delta": {"content": "A"}, "finish_reason": "stop"}], "timings": TIMINGS}


class Tests(unittest.TestCase):
    def delivery(self, root, name="python_tools_long", answer="Created program and tests.", tools=2, before=None):
        records = [{"events": [{"choices": [{"delta": {"content": answer}}]}]}]
        return bench.validate_delivery(name, {"tool_calls": tools}, records, root, before or {})

    def test_reasoning_only_native_success_is_not_delivery(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            result = self.delivery(Path(directory), answer="", tools=0)
            self.assertFalse(result["valid"])
            self.assertIn("Required tool workflow did not occur", result["failures"])

    def test_inherited_artifacts_do_not_count_as_new_work(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            (root / "budget.py").write_text("def main():\n    return 1\n")
            (root / "test_budget.py").write_text("def test_main():\n    assert True\n")
            before = bench.python_snapshot(root)
            self.assertFalse(self.delivery(root, before=before)["valid"])
            self.assertTrue(self.delivery(root)["valid"])
            (root / "budget.py").write_text("def main():\n    pass\n")
            result = self.delivery(root)
            self.assertFalse(result["valid"])
            self.assertIn("Stub function: budget.py:main", result["failures"])

    def test_answer_only_delivery_rejects_tools_or_empty_output(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            self.assertTrue(self.delivery(root, "story", tools=0)["valid"])
            self.assertFalse(self.delivery(root, "story", tools=1)["valid"])
            self.assertFalse(self.delivery(root, "story", tools=0, answer=" \n")["valid"])
            self.assertFalse(self.delivery(root, "python_long", tools=0, answer="No code.")["valid"])

    def test_artifact_symlinks_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            inside = root / "inside"
            inside.mkdir()
            (root / "outside.py").write_text("pass\n")
            (inside / "escaped.py").symlink_to(root / "outside.py")
            with self.assertRaises(ValueError):
                bench.python_snapshot(inside)

    def test_independent_tests_reject_zero_or_failed_tests(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            module = root / "test_delivery.py"
            for body, valid, count in (("import unittest\n", False, 0),
                    ("import unittest\nclass Test(unittest.TestCase):\n    def test_ok(self):\n        self.assertEqual(1, 1)\n", True, 1),
                    ("import unittest\nclass Test(unittest.TestCase):\n    def test_bad(self):\n        self.fail('failure')\n", False, 1)):
                module.write_text(body)
                # Avoid stale bytecode when replacing test source with equal-length fixtures.
                with mock.patch.dict("os.environ", {"PYTHONDONTWRITEBYTECODE": "1"}):
                    result = bench.verify_generated_tests(root, [module.name], root / "tests.log")
                self.assertEqual(result["valid"], valid)
                self.assertEqual(result["test_count"], count)

    def test_independent_tests_do_not_read_stale_workspace_bytecode(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            program = root / "budget.py"
            program.write_text("def value():\n    return 1\n")
            original = program.stat()
            bytecode = Path(py_compile.compile(str(program), doraise=True))
            program.write_text("def value():\n    return 2\n")
            os.utime(program, ns=(original.st_atime_ns, original.st_mtime_ns))
            (root / "test_budget.py").write_text("import unittest, budget\nclass Test(unittest.TestCase):\n    def test_new(self):\n        self.assertEqual(budget.value(), 2)\n")
            result = bench.verify_generated_tests(root, ["test_budget.py"], root / "tests.log")
            self.assertTrue(result["valid"])
            self.assertTrue(bytecode.exists())  # Existing user bytecode is not deleted.

    def test_main_retains_gate_exception_and_runs_next_task(self):
        self.main_failure(False)

    def test_main_retains_unhonored_thinking_override(self):
        self.main_failure(True)

    def test_task_preset_variants_and_wire_guard(self):
        self.main_failure(True, task_presets=True)

    def test_preset_metadata_survives_early_failures(self):
        for stage in ("snapshot", "timeout", "invalid_turn", "native"):
            with self.subTest(stage=stage):
                self.main_failure(True, task_presets=True, failure_stage=stage)

    def main_failure(self, thinking_mismatch, task_presets=False, failure_stage=None):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            argv = ["bench", "--workspace", str(root / "workspace"), "--output", str(root / "output"), "--tasks", "story", "explanation"]
            capture = mock.Mock(records=[{"tag": "story"}, {"tag": "explanation"}])
            if thinking_mismatch:
                if task_presets:
                    argv[1:1] = ["--explicit-sampling", "--task-presets"]
                    capture.records[0]["request"] = bench.task_preset("story")[1] | {"temperature": 0}
                    capture.records[1]["request"] = bench.task_preset("explanation")[1]
                    if failure_stage:
                        capture.records[0]["request"] = bench.task_preset("story")[1]
                else:
                    argv[1:1] = ["--explicit-sampling", "--thinking", "off"]
                    capture.records[0]["request"] = bench.sampling_body(1, "on")
                    capture.records[1]["request"] = bench.sampling_body(1, "off")
            valid = {"valid": True, "test_modules": [], "failures": []}
            delivery = [valid, valid] if thinking_mismatch else [ValueError("escaped symlink"), valid]
            def api(_args, method, path, body=None):
                if method == "post" and path == "/api/session":
                    return {"data": {"id": "fake_session", "location": {"directory": str(root / "workspace")}}}
                if path == "/api/session/active":
                    return {"data": []}
                if path.endswith("/export"):
                    return {"data": {"info": {"outcome": "interrupted" if failure_stage == "native" and capture.tag == "story" else "succeeded"}}}
                return {"data": {}}
            summary_calls = []
            def summary(*_):
                summary_calls.append(1)
                if failure_stage == "invalid_turn" and len(summary_calls) == 1:
                    raise ValueError("invalid model turn")
                return {"turns": [{"error": None, "status": 200}], "outside_model_s": 0, "output_tokens": 2,
                        "decode_tokens": 1, "decode_s": 0.01, "aggregate_decode_tps": 100, "wall_s": 0.02, "end_to_end_tps": 100}
            proc = mock.Mock(returncode=0)
            proc.poll.return_value = 0
            if failure_stage == "timeout":
                proc.wait.side_effect = [subprocess.TimeoutExpired("mock", 1), None]
            with mock.patch("sys.argv", argv), mock.patch.object(bench, "Capture", return_value=capture), \
                    mock.patch.object(bench, "ThreadingHTTPServer"), mock.patch.object(bench, "api", side_effect=api), \
                    mock.patch.object(bench.subprocess, "Popen", return_value=proc) as popen, \
                    mock.patch.object(bench, "summarize", side_effect=summary), \
                    mock.patch.object(bench, "python_snapshot", side_effect=[ValueError("bad snapshot"), {}] if failure_stage == "snapshot" else None, return_value={}), \
                    mock.patch.object(bench, "validate_delivery", side_effect=delivery), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                bench.main()
            results = json.loads((root / "output/results.json").read_text())["results"]
            if failure_stage:
                self.assertTrue(results["story"]["failed"])
                self.assertEqual(results["story"]["model_variant"], "gpu-story")
                self.assertEqual(results["story"]["expected_sampling_body"], bench.task_preset("story")[1])
                self.assertNotIn("failed", results["explanation"])
                return
            self.assertTrue(results["story"]["failed"])
            self.assertEqual(results["story"]["outcome"], "succeeded")
            self.assertIn("sampler body" if thinking_mismatch else "escaped symlink", results["story"]["reason"])
            self.assertNotIn("failed", results["explanation"])
            self.assertEqual(popen.call_count, 2)
            if thinking_mismatch:
                config = json.loads((root / "workspace/opencode.jsonc").read_text())
                model = config["providers"]["qwen-local"]["models"]["local-model"]
                if task_presets:
                    self.assertEqual([v["id"] for v in model["variants"]], ["gpu-story", "gpu-technical"])
                    self.assertIn("qwen-local/local-model#gpu-story", popen.call_args_list[0].args[0])
                    self.assertIn("qwen-local/local-model#gpu-technical", popen.call_args_list[1].args[0])
                    self.assertEqual(results["explanation"]["expected_sampling_body"], bench.task_preset("explanation")[1])
                else:
                    self.assertFalse(model["body"]["chat_template_kwargs"]["enable_thinking"])

    def test_sampling_experiment_validation(self):
        base = ["bench", "--workspace", "/tmp/opencode/unused-validation-workspace", "--output", "/tmp/opencode/unused-validation-output"]
        for extra in (["--thinking", "off"], ["--temperature", "0"], ["--explicit-sampling", "--temperature", "nan"],
                      ["--explicit-sampling", "--temperature", "inf"], ["--explicit-sampling", "--temperature", "2.5"],
                       ["--explicit-sampling", "--temperature", "-1"], ["--task-presets"],
                       ["--explicit-sampling", "--task-presets", "--thinking", "off"],
                       ["--explicit-sampling", "--task-presets", "--temperature", "0.4"]):
            with mock.patch("sys.argv", base + extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                bench.main()
            self.assertEqual(error.exception.code, 2)

    def test_reasoning_experiment_keeps_sampler_defaults(self):
        original = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0,
                    "presence_penalty": 0, "seed": 1234,
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
        self.assertEqual(bench.sampling_body(1.0, "on"), original)
        expected = json.loads(json.dumps(original))
        expected["chat_template_kwargs"]["enable_thinking"] = False
        self.assertEqual(bench.sampling_body(1.0, "off"), expected)

    def test_task_presets_are_distinct_configuration_experiments(self):
        self.assertEqual(bench.task_preset("story"), ("gpu-story", bench.sampling_body(1, "on") | {"reasoning_effort": "xhigh"}))
        self.assertEqual(bench.task_preset("explanation"), ("gpu-technical", bench.sampling_body(0.4, "on") | {"reasoning_effort": "xhigh"}))
        self.assertEqual(bench.task_preset("python_tools_long"), ("gpu-code", bench.sampling_body(0, "off") | {"reasoning_effort": "xhigh"}))

    def test_preset_reasoning_effort_is_checked_on_every_turn(self):
        expected = bench.task_preset("story")[1]
        self.assertTrue(bench.sampling_matches([{"request": expected}], expected))
        for missing_or_wrong in ({k: v for k, v in expected.items() if k != "reasoning_effort"}, expected | {"reasoning_effort": "low"}):
            self.assertFalse(bench.sampling_matches([{"request": expected}, {"request": missing_or_wrong}], expected))

    def test_failed_record_analysis(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "results.json"
            path.write_text(json.dumps({"args": {"tasks": ["python_long"]}, "results": {
                "python_long": {"failed": True, "reason": "task deadline exceeded"}}}))
            output = io.StringIO()
            with mock.patch("sys.argv", ["analyze_client_bench", str(path)]), contextlib.redirect_stdout(output):
                analyze.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["completed_tasks"], [])
            self.assertIsNone(report["completed_only"]["aggregate_decode_tps"])
            self.assertEqual(report["incomplete_or_failed_tasks"], [Path(directory).name + "/python_long"])

    def test_cli_tool_accounting_fallback(self):
        events = [{"type": "tool_use", "part": {"id": "a", "state": {"status": "completed", "time": {"start": 0, "end": 100}}}},
                  {"type": "tool_use", "part": {"id": "b", "state": {"status": "completed", "time": {"start": 50, "end": 200}}}}]
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in events))
            result = bench.cli_tool_metrics(path)
            self.assertIsNone(result["tool_execution_s"])
            self.assertFalse(result["tool_time_valid"])
            self.assertEqual(result["tool_calls"], 2)
            events.append({"type": "tool_use", "part": {"id": "c", "state": {"status": "running"}}})
            path.write_text("\n".join(json.dumps(event) for event in events))
            self.assertFalse(bench.cli_tool_metrics(path)["tool_time_valid"])

    def test_sse(self):
        for ending in [b"\n", b"\r\n", b"\r"]:
            raw = ending.join([b": comment", b"data: {", b'data: "choices":' + json.dumps(EVENT["choices"]).encode() + b",",
                               b'data: "timings":' + json.dumps(TIMINGS).encode(), b"data: }", b"", b"data: [DONE]", b"", b""])
            for size in [1, 2, 7, 9999]:
                with self.subTest(ending=ending, size=size):
                    r = {"index": 0, "request": {}, "started_ns": 0, "events": [], "status": 200}
                    buffer = b""
                    for i in range(0, len(raw), size):
                        buffer = bench.consume_sse(r, buffer, raw[i:i + size])
                    bench.consume_sse(r, buffer, b"", final=True)
                    r["response_ns"] = 1
                    result = bench.summarize([r], 1)
                    self.assertEqual(result["aggregate_decode_tps"], 100)
                    self.assertEqual(result["output_tokens"], 2)
                    self.assertEqual(result["decode_tokens"], 1)

    def test_missing_metrics(self):
        r = {"index": 0, "request": {}, "started_ns": 0, "events": [], "status": 200, "response_ns": 1, "done_ns": 1}
        with self.assertRaises(ValueError):
            bench.summarize([r], 1)

    def test_tool_union(self):
        obj = [{"type": "tool", "id": "a", "state": {}, "time": {"ran": 0, "completed": 100}},
               {"type": "tool", "id": "b", "state": {}, "time": {"ran": 50, "completed": 200}}]
        result = bench.tool_metrics(obj)
        self.assertEqual(result["tool_execution_s"], 0.2)
        self.assertEqual(result["tool_calls"], 2)

    def test_proxy_and_nonstream_replay(self):
        posted = []
        plain = json.dumps({"choices": [{"message": {"content": "A"}}], "timings": TIMINGS}).encode()
        compressed = gzip.compress(plain)
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                posted.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Retry-After", "3")
                self.send_header("Content-Length", str(len(compressed)))
                self.end_headers()
                self.wfile.write(compressed)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
                root = Path(directory)
                capture = bench.Capture(root)
                proxy = ThreadingHTTPServer(("127.0.0.1", 0), bench.handler_factory(capture, server.server_port))
                threading.Thread(target=proxy.serve_forever, daemon=True).start()
                try:
                    body = b'{"stream":false,"messages":[{"role":"user","content":"test"}]}'
                    conn = http.client.HTTPConnection("127.0.0.1", proxy.server_port, timeout=5)
                    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
                    response = conn.getresponse()
                    self.assertEqual(response.getheader("Content-Encoding"), "gzip")
                    self.assertEqual(response.getheader("Retry-After"), "3")
                    self.assertEqual(gzip.decompress(response.read()), plain)
                    conn.close()
                finally:
                    proxy.shutdown(); proxy.server_close()
                self.assertEqual(posted[0], body)
                self.assertEqual(bench.summarize(capture.records, 1)["aggregate_decode_tps"], 100)
                replay = bench.direct_replay(capture.records[0], server.server_port, root, 5)
                self.assertEqual(replay["aggregate_decode_tps"], 100)
                self.assertEqual(json.loads(posted[1]), json.loads(body))
        finally:
            server.shutdown(); server.server_close()

    def test_trickle_deadline_connection_close(self):
        stopped = threading.Event()
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    while not stopped.wait(0.02):
                        self.wfile.write(b": trickle\n\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
                start = time.monotonic()
                with self.assertRaises((TimeoutError, OSError)):
                    bench.direct_replay({"index": 0, "request": {"stream": True}, "path": "/v1/chat/completions"},
                                        server.server_port, Path(directory), 0.3)
                self.assertLess(time.monotonic() - start, 1.0)
        finally:
            stopped.set(); server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
