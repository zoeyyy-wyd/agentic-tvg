#!/usr/bin/env python
"""Re-render LongVT SFT traces to the Qwen3-VL / verl multi-turn format (plan §4.2).

Input: ``longvt_sft_tvg_6k3.parquet`` — 6,395 traces, each exactly 5 messages
with one crop_video call, stored in Qwen2.5-VL-era formatting:
role=user tool responses wrapped in literal ``<tool_response>`` text, a tool
schema carrying a model-supplied ``video_path``, and pre-cropped jpg frames.

Output rows for verl's MultiTurnSFTDataset (messages/images/videos/tools
columns), re-rendered so every surface detail matches what our RL rollout
produces (verified against the Qwen3-VL chat template and ToolAgentLoop):

- system/user turns rebuilt from agentic_tvg.prompts (same as RL data);
  user turn carries the ``<video>`` placeholder -> global 32-frame budget
- assistant tool call: original <think> kept, tool-call block normalized to
  the template's canonical form (json.dumps spacing, no video_path arg)
- tool turn: ``role="tool"`` with <image> placeholders + the exact wording of
  CropVideoTool (crop_response_text) — the template itself adds
  <tool_response> wrappers, so the literal ones from LongVT are dropped
- frames capped at CROP_NUM_FRAMES (uniform subsample, numeric-index order),
  each image entry budgeted with CROP_MAX_PIXELS/CROP_MIN_PIXELS

Usage:
    python data_prep/render_traces.py \
        --traces data/annotations/longvt_sft_tvg_6k3.parquet \
        --video-root data/videos/tvg --out data/processed \
        [--exclude-ids contaminated_ids.txt] [--limit 2000 --seed 0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.constants import (  # noqa: E402
    CROP_MAX_PIXELS,
    CROP_MIN_PIXELS,
    CROP_NUM_FRAMES,
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
)
from agentic_tvg.prompts import build_system_prompt, build_user_prompt  # noqa: E402
from agentic_tvg.span import parse_answer_span  # noqa: E402
from agentic_tvg.crop_video_tool import build_crop_video_schema, crop_response_text  # noqa: E402
from agentic_tvg.video_frames import get_video_duration  # noqa: E402

QUERY_RE = re.compile(r"find the time range of this event\s*:\s*(.+?)\s*\.?\s*Return the time range", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
JPG_IDX_RE = re.compile(r"_(\d+)\.jpg$")


def msg_text(msg) -> str:
    """Concatenate the text segments of a LongVT message."""
    c = msg["content"]
    if isinstance(c, str):
        return c
    return "".join(seg["text"] for seg in c if seg.get("text") is not None)


def msg_images(msg) -> list[str]:
    c = msg["content"]
    if isinstance(c, str):
        return []
    urls = [seg["image_url"]["url"] for seg in c if seg.get("image_url") is not None and seg["image_url"].get("url")]
    # LongVT stores them in string order (0, 1, 10, 11, ...) — re-sort numerically
    return sorted(urls, key=lambda u: int(JPG_IDX_RE.search(u).group(1)) if JPG_IDX_RE.search(u) else 0)


def parse_trace(msgs) -> dict | None:
    """Extract (query, think1, window, jpgs, final) from one 5-message trace."""
    if len(msgs) != 5 or [m["role"] for m in msgs] != ["system", "user", "assistant", "user", "assistant"]:
        return None
    qm = QUERY_RE.search(msg_text(msgs[1]))
    if not qm:
        return None
    a1 = msg_text(msgs[2])
    tc = TOOL_CALL_RE.search(a1)
    if not tc:
        return None
    try:
        args = json.loads(tc.group(1))["arguments"]
        start, end = float(args["start_time"]), float(args["end_time"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    think1 = a1[: tc.start()].rstrip()
    if not (think1.startswith("<think>") and think1.endswith("</think>")):
        return None
    final = msg_text(msgs[4]).strip()
    parsed = parse_answer_span(final)
    if not (parsed.format_ok and parsed.valid):
        return None
    jpgs = msg_images(msgs[3])
    if not jpgs:
        return None
    return {"query": qm.group(1).strip(), "think1": think1, "window": (start, end), "jpgs": jpgs, "final": final, "gt": parsed.span}


def subsample(jpgs: list[str], k: int) -> list[str]:
    if len(jpgs) <= k:
        return jpgs
    idx = np.unique(np.linspace(0, len(jpgs) - 1, k).round().astype(int))
    return [jpgs[i] for i in idx]


def canonical_tool_call(start: float, end: float) -> str:
    """Byte-identical to the Qwen3 chat template's own tool-call rendering."""
    payload = json.dumps({"name": "crop_video", "arguments": {"start_time": start, "end_time": end}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def render(rec: dict, video_path: str, duration: float, jpg_dir: Path) -> dict | None:
    start, end = rec["window"]
    jpgs = [jpg_dir / j for j in subsample(rec["jpgs"], CROP_NUM_FRAMES)]
    if any(not p.exists() for p in jpgs):
        return None
    timestamps = [round(t, 2) for t in np.linspace(start, end, len(jpgs))]

    messages = [
        {"role": "system", "content": build_system_prompt("tool_optional")},
        {"role": "user", "content": build_user_prompt(rec["query"], duration)},
        {"role": "assistant", "content": rec["think1"] + "\n" + canonical_tool_call(start, end)},
        {"role": "tool", "content": "<image>" * len(jpgs) + crop_response_text(start, end, timestamps)},
        {"role": "assistant", "content": rec["final"]},
    ]
    return {
        "messages": messages,
        "images": [
            {"image": str(p), "max_pixels": CROP_MAX_PIXELS, "min_pixels": CROP_MIN_PIXELS} for p in jpgs
        ],
        "videos": [
            {"video": video_path, "nframes": GLOBAL_NUM_FRAMES, "max_pixels": GLOBAL_MAX_PIXELS, "min_pixels": GLOBAL_MIN_PIXELS}
        ],
        "tools": [build_crop_video_schema().model_dump(exclude_unset=True, exclude_none=True)],
        "extra_info": {
            "query": rec["query"],
            "gt": list(rec["gt"]),
            "tool_window": [start, end],
            "duration": duration,
            "n_frames_orig": len(rec["jpgs"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", type=Path, default=Path("data/annotations/longvt_sft_tvg_6k3.parquet"))
    ap.add_argument("--video-root", type=Path, default=Path("data/videos/tvg"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--exclude-ids", type=Path, default=None, help="file with one video id per line to drop (contamination)")
    ap.add_argument("--limit", type=int, default=None, help="subsample N traces (the SFT-dose knob)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    excluded = set()
    if args.exclude_ids and args.exclude_ids.exists():
        excluded = {l.strip() for l in args.exclude_ids.read_text().splitlines() if l.strip()}
        print(f"excluding {len(excluded)} video ids from {args.exclude_ids}")

    df = pd.read_parquet(args.traces)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed).sort_index()

    durations: dict[str, float] = {}
    rows, dropped = [], {"parse": 0, "excluded": 0, "missing_video": 0, "bad_duration": 0, "missing_jpg": 0, "gt_out_of_range": 0}
    for _, r in df.iterrows():
        rec = parse_trace(r["messages"])
        if rec is None:
            dropped["parse"] += 1
            continue
        vid = Path(rec["jpgs"][0]).name.rsplit("_", 3)[0]  # XNT6F_8.6_13.2_0.jpg -> XNT6F
        if vid in excluded:
            dropped["excluded"] += 1
            continue
        video_path = args.video_root / f"{vid}.mp4"
        if not video_path.exists():
            dropped["missing_video"] += 1
            continue
        if vid not in durations:
            try:
                durations[vid] = get_video_duration(str(video_path))
            except Exception:
                durations[vid] = -1.0
        if durations[vid] <= 0:
            dropped["bad_duration"] += 1
            continue
        if rec["gt"][1] > durations[vid] + 1.0 or rec["gt"][0] >= durations[vid]:
            dropped["gt_out_of_range"] += 1
            continue
        row = render(rec, str(video_path.resolve()), durations[vid], args.video_root)
        if row is None:
            dropped["missing_jpg"] += 1
            continue
        row["extra_info"]["video_id"] = vid
        rows.append(row)

    # split by video id so no video straddles train/val
    rng = np.random.default_rng(args.seed)
    vids = sorted({row["extra_info"]["video_id"] for row in rows})
    val_vids = set(rng.choice(vids, size=max(1, int(len(vids) * args.val_frac)), replace=False))
    train = [r for r in rows if r["extra_info"]["video_id"] not in val_vids]
    val = [r for r in rows if r["extra_info"]["video_id"] in val_vids]

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_parquet(args.out / "sft_train.parquet")
    pd.DataFrame(val).to_parquet(args.out / "sft_val.parquet")
    print(f"kept {len(rows)}/{len(df)} (train {len(train)} / val {len(val)}), dropped {dropped}")
    print(f"-> {args.out}/sft_train.parquet, sft_val.parquet")


if __name__ == "__main__":
    main()
