#!/usr/bin/env python3
"""Summarize a decode trace (sqlite export of profile_decode.sh): GPU busy vs idle, time per kernel.

Usage: analyze_nsys.py runs/NAME.sqlite [--top 40]
"""
import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    try:
        db = sqlite3.connect(Path(a.db).resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.OperationalError as e:
        raise SystemExit(f"cannot open {a.db}: {e}")
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"StringIds", "CUPTI_ACTIVITY_KIND_KERNEL"} <= tables:
        raise SystemExit(f"no CUDA kernel trace in {a.db}; was the profile window inside decode?")
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = [(s, e, names[n]) for s, e, n in db.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    copies = list(db.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    graphs = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")) if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else []
    if graphs:
        gt = sum(e - s for s, e in graphs)
        print(f"{len(graphs)} graph launches, {gt / 1e6:.1f} ms total, mean {gt / len(graphs) / 1e3:.1f} us")
    events = sorted([(s, e) for s, e, _ in kernels] + [(s, e) for s, e, _, _ in copies] + graphs)
    if not events:
        raise SystemExit(f"no kernel, memcpy or graph activity in {a.db}; was the profile window inside decode?")
    t0, t1 = events[0][0], max(e for _, e in events)
    span = t1 - t0

    busy, gaps, cur_s, cur_e = 0, [], events[0][0], events[0][1]
    for s, e in events[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s

    print(f"window {span / 1e6:.1f} ms, GPU busy {busy / 1e6:.1f} ms ({100 * busy / span:.1f}%), "
          f"idle {(span - busy) / 1e6:.1f} ms in {len(gaps)} gaps")
    for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e12)):
        sel = [g for g in gaps if lo <= g < hi]
        print(f"  gaps {lo / 1e3:>5.0f}-{hi / 1e3:<8.0f}us: n={len(sel):6d} total={sum(sel) / 1e6:8.1f} ms")

    by_name = defaultdict(lambda: [0, 0])
    for s, e, n in kernels:
        by_name[n][0] += e - s
        by_name[n][1] += 1
    ktotal = sum(v[0] for v in by_name.values())
    print(f"\nkernel time {ktotal / 1e6:.1f} ms, {len(kernels)} launches")
    for n, (t, c) in sorted(by_name.items(), key=lambda x: -x[1][0])[:a.top]:
        print(f"  {100 * t / ktotal:5.1f}% {t / 1e6:8.1f} ms {c:7d}x {t / c / 1e3:8.1f} us  {n[:90]}")
    kinds = defaultdict(lambda: [0, 0, 0])
    for s, e, b, k in copies:
        kinds[k][0] += e - s
        kinds[k][1] += 1
        kinds[k][2] += b
    for k, (t, c, b) in kinds.items():
        print(f"memcpy kind {k}: {c} copies, {b / 1e6:.1f} MB, {t / 1e6:.1f} ms")


if __name__ == "__main__":
    main()
