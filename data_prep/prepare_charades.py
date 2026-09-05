#!/usr/bin/env python
"""Convert Charades-STA (sampled) to the verl val schema, grounding-probe form.

Purpose: an external, short-video (~30s) temporal-grounding comparison across
the recipe stages (sft-mix / grpo-v2 / rft-v2b). The probe does NOT change the
answer format the models were trained on: each row asks a natural "when does X
happen" question, the policy crops where it believes the moment is, and the
metric is the existing evidence_iou -- IoU(crop window, GT span) -- computed by
the same reward code as training. Run with JUDGE_DISABLE=1: there is no QA
ground truth (gt_text is the query sentence), so the acc column is meaningless
and only evidence_iou / num_tool_calls are read. Report mean IoU and R@0.3/0.5.

Annotation line format (charades_sta_test.txt): "VID start end##sentence"

Usage:
  python data_prep/prepare_charades.py \
      --ann data/annotations/charades/sample400.txt \
      --videos-dir data/videos/charades \
      --out data/processed/charades_probe.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))        # data_prep/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # repo root

from extract_rl import build_row  # noqa: E402
from agentic_tvg.video_frames import get_video_duration  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", type=Path, default=Path("data/annotations/charades/sample400.txt"))
    ap.add_argument("--videos-dir", type=Path, default=Path("data/videos/charades"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/charades_probe.parquet"))
    ap.add_argument("--mode", default="tool_optional")   # same prompt mode as every eval
    args = ap.parse_args()

    durations: dict[str, float] = {}
    rows, dropped = [], {"video_missing": 0, "bad_duration": 0, "bad_span": 0}
    for i, line in enumerate(l.strip() for l in open(args.ann) if l.strip()):
        head, sentence = line.split("##", 1)
        vid, start, end = head.split()
        start, end = float(start), float(end)
        vpath = args.videos_dir / f"{vid}.mp4"
        if not vpath.exists():
            dropped["video_missing"] += 1
            continue
        k = str(vpath.resolve())
        if k not in durations:
            try:
                durations[k] = get_video_duration(k)
            except Exception:
                durations[k] = -1.0
        if durations[k] <= 0:
            dropped["bad_duration"] += 1
            continue
        # clamp the GT span to the measured duration; drop degenerate spans
        end = min(end, durations[k])
        if end - start < 0.5:
            dropped["bad_span"] += 1
            continue
        sent = sentence.strip().rstrip(".")
        rows.append(build_row(
            video_path=k, duration=durations[k],
            question=f'During which part of the video does this happen: "{sent}"?',
            gt_text=sent,                      # placeholder; judge is disabled for this probe
            segment=(start, end),              # -> extra_info.video_segment -> evidence_iou
            data_source="charades_sta", split="test",
            index=i, mode=args.mode,
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"{len(rows)} rows -> {args.out} | dropped {dropped}")


if __name__ == "__main__":
    main()
