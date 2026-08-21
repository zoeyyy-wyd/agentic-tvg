#!/usr/bin/env python
"""LongVT-Parquet RL splits -> agentic-TVG train/val parquet (plan §4.1, DATA.md §6.1).

DATA.md §6.1 resolution (verified 2026-08-20 against the downloaded parquet):
the selfqa/rl_val ``reward_model.ground_truth`` is a free-text answer
(style="model"), unusable for IoU — but every row carries
``extra_info.video_segment`` = [start, end], the evidence window. We therefore
recast each QA row as a grounding task: query := the original question,
GT := video_segment. That is exactly the quantity LongVT's own IoU reward
component verifies against.

Requires the videos to be on disk (download_videos.sh): each row's
video is resolved by basename under --video-root, its duration probed with
PyAV, and rows with missing/corrupt videos or out-of-range segments dropped.

Usage (from the repo root, conda env `verl`):
    python data_prep/extract_rl.py \
        --annotations data/annotations --video-root data/videos --out data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.data_schema import build_verl_row, validate_row_against_video  # noqa: E402
from agentic_tvg.video_frames import get_video_duration  # noqa: E402

SPLITS = {
    # annotation file -> (output name, split label)
    "longvt_rl_selfqa_1k6.parquet": ("rl_train.parquet", "train"),
    "longvt_rl_val_114.parquet": ("rl_val.parquet", "val"),
}


def index_videos(video_root: Path) -> dict[str, Path]:
    """basename -> absolute path, over every video file under video_root."""
    exts = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    idx: dict[str, Path] = {}
    for p in video_root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            if p.name in idx:
                print(f"warning: duplicate basename {p.name}: {idx[p.name]} vs {p}")
            idx[p.name] = p.resolve()
    return idx


def convert(src: Path, out: Path, split: str, video_index: dict[str, Path], mode: str) -> None:
    df = pd.read_parquet(src)
    rows, dropped = [], {"missing_video": 0, "bad_duration": 0, "bad_segment": 0}

    for i, r in enumerate(df.itertuples(index=False)):
        info = r.extra_info
        basename = Path(str(r.videos[0]["video"]).replace("file://", "")).name
        path = video_index.get(basename)
        if path is None:
            dropped["missing_video"] += 1
            continue
        try:
            duration = get_video_duration(str(path))
        except Exception:
            dropped["bad_duration"] += 1
            continue
        seg = info.get("video_segment")
        gt = validate_row_against_video((float(seg[0]), float(seg[1])), duration) if seg is not None and len(seg) == 2 else None
        if gt is None:
            dropped["bad_segment"] += 1
            continue
        rows.append(
            build_verl_row(
                video_path=str(path),
                duration=duration,
                query=str(info["question"]),
                gt_span=gt,
                data_source=f"longvt_{r.data_source}",
                split=split,
                index=i,
                mode=mode,
                extra={"orig_answer": str(info.get("answer", ""))},
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"{src.name}: kept {len(rows)}/{len(df)}, dropped {dropped} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    ap.add_argument("--video-root", type=Path, default=Path("data/videos"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--mode", default="tool_optional", help="system prompt mode baked into the rows")
    args = ap.parse_args()

    video_index = index_videos(args.video_root)
    print(f"indexed {len(video_index)} videos under {args.video_root}")
    for src_name, (out_name, split) in SPLITS.items():
        convert(args.annotations / src_name, args.out / out_name, split, video_index, args.mode)


if __name__ == "__main__":
    main()
