"""Builders for verl-schema parquet rows (RL train / val / eval).

One row layout shared by every dataset we emit, matching what verl 0.9.0's
RLHFDataset + ToolAgentLoop consume (verified against the source):

- ``prompt``: chat messages; ``<video>`` in the user turn is replaced by the
  entry from ``videos``.
- ``videos``: one dict per placeholder; extra keys (nframes, min/max_pixels)
  are forwarded to qwen-vl-utils, which is how the global coarse view budget
  is enforced.
- ``reward_model.ground_truth``: [start, end] seconds — the RLVR signal.
- ``extra_info.tools_kwargs.crop_video.create_kwargs``: per-trajectory state
  handed to CropVideoTool.create().
- ``agent_name``: routes the row to ToolAgentLoop.
"""

from __future__ import annotations

from typing import Any

from agentic_tvg.constants import (
    GLOBAL_MAX_PIXELS,
    GLOBAL_MIN_PIXELS,
    GLOBAL_NUM_FRAMES,
    TOOL_NAME,
)
from agentic_tvg.prompts import build_system_prompt, build_user_prompt


def build_verl_row(
    *,
    video_path: str,
    duration: float,
    query: str,
    gt_span: tuple[float, float],
    data_source: str,
    split: str,
    index: int,
    mode: str = "tool_optional",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One TVG sample in verl schema. ``video_path`` must be absolute."""
    gt = [round(float(gt_span[0]), 3), round(float(gt_span[1]), 3)]
    row = {
        "data_source": data_source,
        "agent_name": "tool_agent",
        "prompt": [
            {"role": "system", "content": build_system_prompt(mode)},
            {"role": "user", "content": build_user_prompt(query, duration)},
        ],
        "videos": [
            {
                "type": "video",
                "video": video_path,
                "nframes": GLOBAL_NUM_FRAMES,
                "max_pixels": GLOBAL_MAX_PIXELS,
                "min_pixels": GLOBAL_MIN_PIXELS,
            }
        ],
        "ability": "temporal_video_grounding",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": index,
            "question": query,
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
    if extra:
        row["extra_info"].update(extra)
    return row


def validate_row_against_video(gt: tuple[float, float], duration: float) -> tuple[float, float] | None:
    """Sanity-check a GT span against the real video duration.

    Returns a (possibly end-clamped) span, or None if the sample must be
    dropped (empty span, or span starting at/after EOF — annotation offset
    errors would silently corrupt the IoU reward otherwise).
    """
    s, e = float(gt[0]), float(gt[1])
    if e <= s or s < 0 or s >= duration:
        return None
    return (s, min(e, duration))
