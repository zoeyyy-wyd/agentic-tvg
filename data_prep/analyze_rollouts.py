#!/usr/bin/env python
"""Row-level analysis of validation/benchmark rollout jsonl(s), as markdown.

Where score_rollouts.py stops at means, this answers "what actually changed
and where is the headroom":

- error taxonomy: acc (judge FULL/PARTIAL/WRONG) crossed with whether the
  policy's crop window hit the GT evidence (evidence_iou >= 0.3, the same
  threshold the RFT data filter used). Splits "found the moment but misread
  it" (perception) from "never found it" (localization) -- the two need
  different fixes.
- tool-call distribution and response length per side.
- with --compare: the flipped rows, question by question, with both answers
  side by side -- the paired detail a mean difference hides.

Benchmark rows (videosiah_eval) have no GT window, so their taxonomy
degenerates to the acc column alone; the script says so rather than printing
a misleading all-zero IoU column.

Usage:
  python data_prep/analyze_rollouts.py results/val-rft/val_rollouts/0.jsonl \
      --compare results/grpo-vanilla/val_rollouts_grpo267/267.jsonl \
      --labels RFT GRPO -o results/val-rft/analysis.md
  python data_prep/analyze_rollouts.py results/bench-rft/chunk_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_tvg.answer_match import parse_answer_qa  # noqa: E402
from agentic_tvg.reward import _crop_windows  # noqa: E402

# build_user_prompt embeds these exact strings (prompts.py, byte-frozen).
_Q_RE = re.compile(r'The video is ([\d.]+) seconds long\. Question: "(.*)"\s*\n'
                   r"Answer the question based on the video", re.DOTALL)

IOU_GROUNDED = 0.3   # the RFT-filter threshold; below it the crop missed the evidence


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    for r in rows:
        m = _Q_RE.search(r.get("input", ""))
        r["_question"] = m.group(2).strip() if m else "(question not parsed)"
        r["_duration"] = float(m.group(1)) if m else None
        r["_answer"] = parse_answer_qa(r.get("output", "")).answer or "(no answer tag)"
        r["_windows"] = _crop_windows(r.get("output", ""))
    return rows


def fmt_windows(ws: list[tuple[float, float]]) -> str:
    return " ".join(f"[{s:.0f},{e:.0f}]" for s, e in ws) or "(no crop)"


def mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return sum(vals) / len(vals) if vals else float("nan")


def has_gt_windows(rows: list[dict]) -> bool:
    # benchmark rows carry no GT segment; their evidence_iou is identically 0
    # by construction, which is indistinguishable from "always missed" -- only
    # treat IoU as meaningful when some row scored above 0.
    return any(float(r.get("evidence_iou", 0.0)) > 0.0 for r in rows)


def section_aggregate(out: list[str], sides: list[tuple[str, list[dict]]]) -> None:
    metrics = ["format_score", "answered", "acc", "evidence_iou", "num_tool_calls", "score"]
    out.append("## Aggregate\n")
    out.append("| | " + " | ".join(lbl for lbl, _ in sides) + " |")
    out.append("|---|" + "---:|" * len(sides))
    out.append("| n | " + " | ".join(str(len(r)) for _, r in sides) + " |")
    for m in metrics:
        out.append(f"| {m} | " + " | ".join(f"{mean(r, m):.4f}" for _, r in sides) + " |")
    for lbl, rows in sides:
        dist = {k: sum(1 for r in rows if int(float(r.get("num_tool_calls", 0))) == k)
                for k in range(4)}
        chars = sum(len(r.get("output", "")) for r in rows) / max(len(rows), 1)
        out.append(f"\n{lbl}: tool calls 0/1/2/3 = "
                   f"{dist[0]}/{dist[1]}/{dist[2]}/{dist[3]} rows, "
                   f"mean response {chars:.0f} chars")
    out.append("")


def section_taxonomy(out: list[str], sides: list[tuple[str, list[dict]]]) -> None:
    out.append("## Error taxonomy (acc x evidence window)\n")
    for lbl, rows in sides:
        if not has_gt_windows(rows):
            counts = {a: sum(1 for r in rows if float(r["acc"]) == a) for a in (1.0, 0.5, 0.0)}
            out.append(f"{lbl}: no GT windows in this set (benchmark rows) -- acc only: "
                       f"FULL {counts[1.0]}, PARTIAL {counts[0.5]}, WRONG {counts[0.0]}\n")
            continue
        out.append(f"**{lbl}** (grounded = evidence_iou >= {IOU_GROUNDED}):\n")
        out.append("| judge | grounded | not grounded | reading |")
        out.append("|---|---:|---:|---|")
        note = {1.0: "correct", 0.5: "partial", 0.0: "wrong"}
        hint = {1.0: "right window -> right answer / lucky prior",
                0.5: "",
                0.0: "perception miss <-> localization miss"}
        for a in (1.0, 0.5, 0.0):
            g = sum(1 for r in rows if float(r["acc"]) == a
                    and float(r["evidence_iou"]) >= IOU_GROUNDED)
            ng = sum(1 for r in rows if float(r["acc"]) == a
                     and float(r["evidence_iou"]) < IOU_GROUNDED)
            out.append(f"| {note[a]} | {g} | {ng} | {hint[a]} |")
        out.append("")


def section_flips(out: list[str], a_lbl: str, a: list[dict],
                  b_lbl: str, b: list[dict], max_rows: int) -> None:
    bk = {r["input"]: r for r in b if "input" in r}
    paired = [(r, bk[r["input"]]) for r in a if r.get("input") in bk]
    out.append(f"## Flipped rows ({a_lbl} vs {b_lbl}, paired on prompt: {len(paired)})\n")
    for title, keep in [(f"{a_lbl} better", lambda x, y: float(x["acc"]) > float(y["acc"])),
                        (f"{a_lbl} worse", lambda x, y: float(x["acc"]) < float(y["acc"]))]:
        flips = [(x, y) for x, y in paired if keep(x, y)]
        out.append(f"### {title}: {len(flips)} rows\n")
        for x, y in flips[:max_rows]:
            out.append(f"- **Q:** {x['_question'][:220]}")
            out.append(f"  - GT: {str(x.get('gts', ''))[:160]}")
            out.append(f"  - {a_lbl}: acc {float(x['acc']):.1f}, iou {float(x['evidence_iou']):.2f}, "
                       f"crop {fmt_windows(x['_windows'])} -> {x['_answer'][:160]}")
            out.append(f"  - {b_lbl}: acc {float(y['acc']):.1f}, iou {float(y['evidence_iou']):.2f}, "
                       f"crop {fmt_windows(y['_windows'])} -> {y['_answer'][:160]}")
        if len(flips) > max_rows:
            out.append(f"  ... {len(flips) - max_rows} more")
        out.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--compare", nargs="+", metavar="JSONL")
    ap.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("A", "B"))
    ap.add_argument("--max-flips", type=int, default=40, help="cap per flip section")
    ap.add_argument("-o", "--out", type=Path, help="write markdown here instead of stdout")
    args = ap.parse_args()

    a = load(args.jsonl)
    sides = [(args.labels[0], a)]
    if args.compare:
        sides.append((args.labels[1], load(args.compare)))

    out: list[str] = [f"# Rollout analysis: {' vs '.join(lbl for lbl, _ in sides)}\n"]
    section_aggregate(out, sides)
    section_taxonomy(out, sides)
    if args.compare:
        section_flips(out, args.labels[0], a, args.labels[1], sides[1][1], args.max_flips)

    text = "\n".join(out) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    else:
        print(text)


if __name__ == "__main__":
    main()
