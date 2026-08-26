#!/usr/bin/env python
"""Survey every LongVT annotation split: task shape, trace shape, video length.

Answers "which splits are long-video temporal grounding?" from the 172 MB
annotation download alone -- no video needed. Durations are read from
extra_info when present (video_segment endpoints give a lower bound on
duration even when duration itself is absent).

Usage:  python data_prep/inspect_splits.py [--annotations data/annotations]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def probe_extra_info(df: pd.DataFrame) -> dict:
    """Pull duration / span hints out of extra_info, whatever its encoding."""
    if "extra_info" not in df.columns:
        return {}
    out: dict[str, list] = {"duration": [], "span_end": [], "span_len": []}
    for v in df["extra_info"].head(400):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                continue
        if not isinstance(v, dict):
            continue
        for key in ("duration", "video_duration"):
            if isinstance(v.get(key), (int, float)):
                out["duration"].append(float(v[key]))
                break
        seg = next((v[k] for k in ("video_segment", "gt", "timestamps")
                    if v.get(k) is not None), None)   # `or` chain breaks on numpy arrays
        if isinstance(seg, (list, tuple, np.ndarray)) and len(seg) == 2:
            try:
                s, e = float(seg[0]), float(seg[1])
                out["span_end"].append(e)
                out["span_len"].append(e - s)
            except (TypeError, ValueError):
                pass
    return {k: v for k, v in out.items() if v}


def describe(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    return f"min {a.min():.0f} / med {np.median(a):.0f} / p95 {np.percentile(a, 95):.0f} / max {a.max():.0f}"


def trace_shape(df: pd.DataFrame) -> str:
    if "messages" not in df.columns:
        return "no messages column"
    lens, tool_calls = [], 0
    for msgs in df["messages"].head(200):
        try:
            lens.append(len(msgs))
            if any("<tool_call>" in json.dumps(m.get("content"), default=str) for m in msgs):
                tool_calls += 1
        except (TypeError, AttributeError):
            pass
    if not lens:
        return "unreadable messages"
    return f"{min(lens)}-{max(lens)} msgs, tool_call in {tool_calls}/{len(lens)} sampled"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    args = ap.parse_args()

    files = sorted(args.annotations.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.annotations} -- run prepare_data.sh first")

    for f in files:
        df = pd.read_parquet(f)
        print(f"\n\033[1m{f.name}\033[0m  rows={len(df)}")
        print(f"  columns: {list(df.columns)}")
        print(f"  traces : {trace_shape(df)}")
        hints = probe_extra_info(df)
        if "duration" in hints:
            print(f"  \033[1;32mduration (s): {describe(hints['duration'])}\033[0m")
        elif "span_end" in hints:
            print(f"  span end (s, >= lower bound on duration): {describe(hints['span_end'])}")
        if "span_len" in hints:
            print(f"  GT span length (s): {describe(hints['span_len'])}")


if __name__ == "__main__":
    main()
