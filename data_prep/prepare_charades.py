#!/usr/bin/env python
"""Charades-STA test split -> eval parquet in verl schema (plan §4.1 / §7).

Inputs (both from official sources, DATA.md §4):
- ``charades_sta_test.txt`` — Charades-STA annotations (Gao et al., TALL),
  one line per sample: ``<video_id> <start> <end>##<sentence>``
- ``Charades_v1_480.zip`` extracted under --video-root (AllenAI:
  https://prior.allenai.org/projects/charades — Data (scaled to 480p))

Durations are probed per video with PyAV; annotations pointing past EOF are
end-clamped, degenerate ones dropped (Charades-STA has a handful of both).

Usage:
    python data_prep/prepare_charades.py \
        --sta-file data/annotations/charades_sta_test.txt \
        --video-root data/videos/charades --out data/processed/charades_test.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.data_schema import build_verl_row, validate_row_against_video  # noqa: E402
from agentic_tvg.video_frames import get_video_duration  # noqa: E402


def parse_sta_line(line: str) -> tuple[str, float, float, str] | None:
    line = line.strip()
    if not line or "##" not in line:
        return None
    head, sentence = line.split("##", 1)
    parts = head.split()
    if len(parts) != 3:
        return None
    vid, s, e = parts[0], float(parts[1]), float(parts[2])
    return vid, s, e, sentence.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sta-file", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/charades_test.parquet"))
    ap.add_argument("--mode", default="tool_optional")
    args = ap.parse_args()

    # index videos once (Charades_v1_480 is flat: <id>.mp4)
    videos = {p.stem: p.resolve() for p in args.video_root.rglob("*.mp4")}
    print(f"indexed {len(videos)} videos under {args.video_root}")

    durations: dict[str, float] = {}
    rows, dropped = [], {"unparsable": 0, "missing_video": 0, "bad_duration": 0, "bad_segment": 0}
    for i, line in enumerate(args.sta_file.read_text().splitlines()):
        parsed = parse_sta_line(line)
        if parsed is None:
            if line.strip():
                dropped["unparsable"] += 1
            continue
        vid, s, e, sentence = parsed
        path = videos.get(vid)
        if path is None:
            dropped["missing_video"] += 1
            continue
        if vid not in durations:
            try:
                durations[vid] = get_video_duration(str(path))
            except Exception:
                durations[vid] = -1.0
        if durations[vid] <= 0:
            dropped["bad_duration"] += 1
            continue
        gt = validate_row_against_video((s, e), durations[vid])
        if gt is None:
            dropped["bad_segment"] += 1
            continue
        rows.append(
            build_verl_row(
                video_path=str(path),
                duration=durations[vid],
                query=sentence,
                gt_span=gt,
                data_source="charades_sta",
                split="test",
                index=i,
                mode=args.mode,
                extra={"video_id": vid},
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"kept {len(rows)} samples, dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()
