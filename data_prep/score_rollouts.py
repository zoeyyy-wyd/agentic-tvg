#!/usr/bin/env python
"""Mean metrics over validation/benchmark rollout jsonl(s), README-table style.

Works on anything _validate dumps (validation_data_dir): each line carries the
per-row scores from compute_score_qa (acc/format_score/evidence_iou/...).

Usage:
  python data_prep/score_rollouts.py results/val-rft/val_rollouts/0.jsonl
  python data_prep/score_rollouts.py results/bench-rft/chunk_*.jsonl
  python data_prep/score_rollouts.py results/val-rft/val_rollouts/0.jsonl \
      --compare results/grpo-vanilla/val_rollouts_grpo267/267.jsonl

--compare pairs rows across the two sides by their full prompt text (same val
set => same prompts), prints side-by-side means + delta, and counts rows whose
acc improved/regressed -- the paired view that a mean difference hides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["format_score", "answered", "acc", "evidence_iou", "num_tool_calls", "score"]


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def means(rows: list[dict]) -> dict[str, float]:
    out = {"n": float(len(rows))}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if m in r]
        out[m] = sum(vals) / len(vals) if vals else float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--compare", nargs="+", metavar="JSONL",
                    help="second side; paired on prompt text")
    args = ap.parse_args()

    a = load(args.jsonl)
    ma = means(a)
    if not args.compare:
        print(f"n = {len(a)}   ({', '.join(Path(p).name for p in args.jsonl[:4])}"
              f"{' ...' if len(args.jsonl) > 4 else ''})")
        for m in METRICS:
            print(f"  {m:15s} {ma[m]:.4f}")
        return

    b = load(args.compare)
    mb = means(b)
    print(f"{'':15s} {'A':>8s} {'B':>8s} {'Δ (A-B)':>9s}      A={len(a)} rows, B={len(b)} rows")
    for m in METRICS:
        print(f"  {m:15s} {ma[m]:8.4f} {mb[m]:8.4f} {ma[m]-mb[m]:+9.4f}")

    bk = {r["input"]: r for r in b if "input" in r}
    paired = [(r, bk[r["input"]]) for r in a if r.get("input") in bk]
    if paired:
        up = sum(1 for x, y in paired if float(x["acc"]) > float(y["acc"]))
        down = sum(1 for x, y in paired if float(x["acc"]) < float(y["acc"]))
        print(f"paired on prompt: {len(paired)} rows | acc up {up}, down {down}, "
              f"unchanged {len(paired) - up - down}")
    else:
        print("paired on prompt: 0 rows matched (different val sets?)")


if __name__ == "__main__":
    main()
