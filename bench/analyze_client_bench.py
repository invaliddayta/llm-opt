#!/usr/bin/env python3
"""Report completed-task rates and exact-output-matched client/API replay rates."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

from opencode_client_bench import consume_sse, validate_record


def generation(path):
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if raw.lstrip()[:1] == b"{":
        events = [{"choices": [choice | {"delta": choice.get("message", {})} for choice in json.loads(raw).get("choices", [])]}]
    else:
        record = {"events": []}
        buffer = b""
        for i in range(0, len(raw), 65536):
            buffer = consume_sse(record, buffer, raw[i:i + 65536])
        consume_sse(record, buffer, b"", final=True)
        events = record["events"]
    parts = defaultdict(str)
    for event in events:
        for choice in event.get("choices", []):
            prefix = str(choice.get("index", 0))
            delta = choice.get("delta") or {}
            for field in ("content", "reasoning_content"):
                if isinstance(delta.get(field), str):
                    parts[prefix + ":" + field] += delta[field]
            for i, call in enumerate(delta.get("tool_calls") or []):
                for field, text in (call.get("function") or {}).items():
                    if isinstance(text, str):
                        parts[prefix + ":tool:" + str(call.get("index", i)) + ":" + field] += text
    if not parts:
        raise ValueError(f"No generated text or tool calls in {path}")
    return hashlib.sha256(json.dumps(dict(parts), sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    completed = []
    missing = []
    replay_details = []
    for path in args.results:
        data = json.loads(path.read_text())
        root = path.parent
        missing.extend(f"{root.name}/{name}" for name in data["args"]["tasks"] if name not in data["results"])
        for name, result in data["results"].items():
            delivered = result.get("delivery", {}).get("valid", True)
            tested = result.get("independent_tests", {}).get("valid", True)
            if result.get("failed") or result.get("returncode") != 0 or result.get("outcome", "succeeded") != "succeeded" or not delivered or not tested:
                missing.append(f"{root.name}/{name}")
                continue
            for turn in result["turns"]:
                validate_record(json.loads((root / f"wire-{turn['wire_index']:03d}.json").read_text()))
            completed.append({"source": root.name, "task": name, "output_tokens": result["output_tokens"],
                              "decode_tokens": result["decode_tokens"], "decode_s": result["decode_s"],
                              "decode_tps": result["aggregate_decode_tps"], "wall_s": result["wall_s"],
                              "tool_execution_s": result.get("tool_execution_s")})
            for original, replay in zip(result["turns"], result.get("direct_replay", [])):
                index = original["wire_index"]
                a = generation(root / f"wire-{index:03d}.response")
                b = generation(root / f"replay-{index:03d}.response")
                replay_details.append({"source": root.name, "task": name, "wire_index": index,
                    "generation_equal": a == b, "original_sha256": a, "replay_sha256": b,
                    "decode_tokens": original["timings"]["predicted_n"] - 1,
                    "opencode_decode_s": original["timings"]["predicted_ms"] / 1000,
                    "direct_decode_s": replay["decode_s"]})
    n = sum(r["decode_tokens"] for r in completed)
    sec = sum(r["decode_s"] for r in completed)
    matched = [r for r in replay_details if r["generation_equal"]]
    matched_n = sum(r["decode_tokens"] for r in matched)
    report = {"completed_tasks": completed, "incomplete_or_failed_tasks": missing,
              "completed_only": {"aggregate_decode_tps": n / sec if sec else None,
                  "mean_task_decode_tps": statistics.mean(r["decode_tps"] for r in completed) if completed else None,
                  "output_tokens": sum(r["output_tokens"] for r in completed), "decode_tokens": n},
              "replay": {"turns": len(replay_details), "exact_generation_matches": len(matched),
                  "matched_decode_tokens": matched_n,
                  "matched_opencode_decode_tps": matched_n / sum(r["opencode_decode_s"] for r in matched) if matched else None,
                  "matched_direct_decode_tps": matched_n / sum(r["direct_decode_s"] for r in matched) if matched else None,
                  "details": replay_details}}
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to replace {args.output}")
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "replay"} | {
        "replay": {k: v for k, v in report["replay"].items() if k != "details"}}, indent=2))


if __name__ == "__main__":
    main()
