#!/usr/bin/env python
"""Build the RL parquets for GRPO (PLAN.md 3/5, provenance DATA.md 1).

- rl_train.parquet: the RL pool from qa_allocation -- selfqa questions that are
  question/video-disjoint from SFT and whose answers are matcher-verifiable
  (<= --max-gt-words). 893 upper bound; rows whose videos are missing/corrupt
  are dropped here, difficulty filtering happens later.
- rl_val.parquet:  longvt_rl_val_114 verbatim (zero overlap with training).

Every row is verl RLHFDataset + ToolAgentLoop schema (prompt/videos/
reward_model/extra_info/agent_name), with:
- prompt rebuilt from agentic_tvg.prompts (QA mode) -- byte-identical to the
  SFT rows, per the re-rendering discipline (PLAN.md 4)
- reward_model.ground_truth = the raw GT text (the judge grades it; the
  no-key offline fallback expands rule aliases at scoring time)
- extra_info.video_segment = the evidence window for the R_time reward term

Usage:  python data_prep/extract_rl.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_tvg.constants import (  # noqa: E402
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
    TOOL_NAME,
)
from agentic_tvg.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from agentic_tvg.video_frames import get_video_duration  # noqa: E402

VID_RE = re.compile(r"([A-Za-z0-9_\-]+\.mp4)")


def build_row(*, video_path: str, duration: float, question: str, gt_text: str,
              segment: tuple[float, float] | None, data_source: str, split: str,
              index: int, mode: str) -> dict:
    """One QA sample in verl schema (layout verified for verl 0.9.0 agent loop)."""
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": build_system_prompt(mode)},
            {"role": "user", "content": build_user_prompt(question, duration)},
        ],
        "videos": [{
            "type": "video",
            "video": video_path,
            "nframes": GLOBAL_NUM_FRAMES,
            "max_pixels": GLOBAL_MAX_PIXELS,
            "min_pixels": GLOBAL_MIN_PIXELS,
        }],
        "ability": "video_qa_tool",
        "reward_model": {"style": "rule", "ground_truth": gt_text},   # raw text; the judge grades it, the offline fallback expands aliases on the fly
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "gt_text": gt_text,
            "video_segment": [round(segment[0], 3), round(segment[1], 3)] if segment else None,
            "duration": round(float(duration), 3),
            "video_path": video_path,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                TOOL_NAME: {
                    "create_kwargs": {"video_path": video_path, "duration": round(float(duration), 3)},
                },
            },
        },
    }


def clamp_segment(seg, duration: float) -> tuple[float, float] | None:
    if seg is None or len(seg) != 2:
        return None
    s, e = float(seg[0]), float(seg[1])
    if e <= s or s < 0 or s >= duration:
        return None
    return (s, min(e, duration))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allocation", type=Path, default=Path("data/processed/allocation.json"))
    ap.add_argument("--selfqa", type=Path, default=Path("data/annotations/longvt_rl_selfqa_1k6.parquet"))
    ap.add_argument("--rl-val", type=Path, default=Path("data/annotations/longvt_rl_val_114.parquet"))
    ap.add_argument("--selfqa-video-root", type=Path, default=Path("data/videos/selfqa"))
    ap.add_argument("--rl-val-video-root", type=Path, default=Path("data/videos/rl_val"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--mode", default="tool_optional")
    args = ap.parse_args()

    rl_questions = {x["question"] for x in json.loads(args.allocation.read_text())["rl"]}

    durations: dict[str, float] = {}

    def duration_of(path: Path) -> float:
        k = str(path)
        if k not in durations:
            try:
                durations[k] = get_video_duration(k)
            except Exception:
                durations[k] = -1.0
        return durations[k]

    for src, root, out_name, split, keep in [
        (args.selfqa, args.selfqa_video_root, "rl_train.parquet", "train",
         lambda q: q in rl_questions),
        (args.rl_val, args.rl_val_video_root, "rl_val.parquet", "val",
         lambda q: True),
    ]:
        df = pd.read_parquet(src)
        rows, dropped = [], {"not_in_pool": 0, "missing_video": 0, "bad_duration": 0, "bad_segment": 0}
        for i, r in df.iterrows():
            ei = dict(r["extra_info"])
            q = ei["question"].strip()
            if not keep(q):
                dropped["not_in_pool"] += 1
                continue
            m = VID_RE.search(str(list(r["videos"])))
            vpath = (root / m.group(1)).resolve() if m else None
            if vpath is None or not vpath.exists():
                dropped["missing_video"] += 1
                continue
            dur = duration_of(vpath)
            if dur <= 0:
                dropped["bad_duration"] += 1
                continue
            seg = clamp_segment(ei.get("video_segment"), dur)
            if seg is None:
                dropped["bad_segment"] += 1
                continue
            gt_text = dict(r["reward_model"])["ground_truth"].strip()
            rows.append(build_row(
                video_path=str(vpath), duration=dur, question=q, gt_text=gt_text,
                segment=seg, data_source=f"longvt_{r['data_source']}",
                split=split, index=int(i), mode=args.mode,
            ))
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(args.out / out_name)
        print(f"{src.name}: kept {len(rows)}/{len(df)}, dropped {dropped} -> {args.out / out_name}")


if __name__ == "__main__":
    main()
