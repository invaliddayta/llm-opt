#!/usr/bin/env python3
"""Request-compatibility smoke test for an OpenAI-compatible llama-server.

Covers requests that the GPU sampler cannot serve (they must fall back to the
standard sampler, not fail), tool calls, an image, and a final plain request
to check that GPU sampling resumes. Exit code 0 means every check passed.

Usage: compat_check.py [--url http://127.0.0.1:8080] [--model local-model]
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}, time.time() - t0


def red_png(size=32):
    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="local-model")
    a = ap.parse_args()

    def base(content, **extra):
        body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": 400,
                "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
        body.update(extra)
        return body

    image = "data:image/png;base64," + base64.b64encode(red_png()).decode()
    cases = [
        ("plain", base("What is 6*7? Answer with the number only."),
         lambda d: "42" in d["choices"][0]["message"]["content"]),
        ("presence_penalty", base("What is 6*7? Answer with the number only.", presence_penalty=1.5),
         lambda d: "42" in d["choices"][0]["message"]["content"]),
        ("repeat_penalty", base("List three colors.", repeat_penalty=1.1, frequency_penalty=0.2),
         lambda d: len(d["choices"][0]["message"]["content"]) > 0),
        ("logprobs", base("Say yes.", logprobs=True, top_logprobs=3, max_tokens=8),
         lambda d: d["choices"][0].get("logprobs") is not None),
        ("json_schema", base("Give a person named Ann aged 30 as JSON.", response_format={"type": "json_schema", "json_schema": {
            "name": "p", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                                     "required": ["name", "age"]}}}),
         lambda d: json.loads(d["choices"][0]["message"]["content"])["age"] == 30),
        ("tool_call", base("What's the weather in Berlin? Use the tool.", tools=[WEATHER_TOOL]),
         lambda d: d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"),
        ("tool_call_thinking", dict(base("What's the weather in Paris? Use the tool.", tools=[WEATHER_TOOL], max_tokens=2000),
                                    chat_template_kwargs={"enable_thinking": True}),
         lambda d: "Paris" in d["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]),
        ("image", base([{"type": "text", "text": "What color is this image? One word."},
                        {"type": "image_url", "image_url": {"url": image}}]),
         lambda d: "red" in d["choices"][0]["message"]["content"].lower()),
        ("plain_after", base("What is 7*8? Answer with the number only."),
         lambda d: "56" in d["choices"][0]["message"]["content"]),
    ]
    failed = 0
    for name, body, check in cases:
        status, data, wall = post(a.url, body)
        ok = False
        detail = ""
        if status == 200:
            try:
                ok = bool(check(data))
            except Exception as e:
                detail = f"check error: {e!r}"
            tps = (data.get("timings") or {}).get("predicted_per_second")
            detail = detail or f"tps={tps:.1f}" if tps else detail
        else:
            detail = str(data)[:300]
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:20s} http={status} {wall:6.1f}s {detail}", flush=True)
        if not ok and status == 200:
            print("   ", json.dumps(data)[:400])
    print(f"{len(cases) - failed}/{len(cases)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
