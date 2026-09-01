#!/usr/bin/env python
"""Convert VideoSIAH-Eval (longvideotool/VideoSIAH-Eval) to the verl val schema.

The benchmark is 652 open-ended QA rows (video_path/question/answer) over 244
videos. There is NO ground-truth time window, so extra_info.video_segment is
None and the evidence_iou reward term is identically 0 here: this benchmark
measures answer accuracy only. That matches the paper -- LongVT scores
VideoSIAH-Eval on answers and runs its temporal-IoU ablation on Charades-STA.

Only rows whose video exists under --videos-dir are emitted. That is the
chunking contract with run_benchmark.sh: the full video set is ~109G and does
not fit on this disk, so the driver downloads one ~10G zip at a time and calls
this script to parquet whatever is currently on disk. Videos are matched by
basename anywhere under --videos-dir (zip layouts differ per chunk).

Usage:
  python data_prep/prepare_videosiah_eval.py \
      --videos-dir data/videos/videosiah_eval \
      --out data/processed/videosiah_chunk.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))        # data_prep/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # repo root

from extract_rl import build_row  # noqa: E402  (also pulls agentic_tvg onto the path)
from agentic_tvg.video_frames import get_video_duration  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", type=Path,
                    default=Path("data/annotations/videosiah_eval/data/test-00000-of-00001.parquet"))
    ap.add_argument("--videos-dir", type=Path, default=Path("data/videos/videosiah_eval"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/videosiah_chunk.parquet"))
    ap.add_argument("--mode", default="tool_optional")   # same prompt mode as rl_val
    args = ap.parse_args()

    df = pd.read_parquet(args.qa)
    by_name = {p.name: p for p in sorted(args.videos_dir.rglob("*.mp4"))}

    durations: dict[str, float] = {}
    rows, dropped = [], {"video_not_local": 0, "bad_duration": 0}
    for i, r in df.iterrows():
        vpath = by_name.get(Path(str(r["video_path"])).name)
        if vpath is None:
            dropped["video_not_local"] += 1
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
        rows.append(build_row(
            video_path=k, duration=durations[k],
            question=str(r["question"]).strip(), gt_text=str(r["answer"]).strip(),
            segment=None, data_source="videosiah_eval", split="test",
            index=int(i),   # stable id in the 652-row benchmark parquet, for joins
            mode=args.mode,
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["data_source", "agent_name", "prompt", "videos", "ability", "reward_model", "extra_info"]
    pd.DataFrame(rows, columns=cols).to_parquet(args.out)
    print(f"{args.qa.name}: kept {len(rows)}/{len(df)} "
          f"({len(by_name)} local videos), dropped {dropped} -> {args.out}")


if __name__ == "__main__":
    main()
