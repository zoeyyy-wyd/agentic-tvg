"""crop_video tool for verl's ToolAgentLoop (plan §2).

Registered via ``crop_video_tool.yaml (repo root)``. Per-trajectory state
(video path, duration) arrives through the dataset row's
``extra_info.tools_kwargs.crop_video.create_kwargs`` — the model itself only
passes (start_time, end_time), so it can never point the tool at a wrong file.

Returns ToolResponse(image=[PIL...], text=...). ToolAgentLoop renders the
image block *before* the text block in the tool message, so the text refers
back to "the frames above" and carries their timestamps.

NOTE: returning a *list* of images requires the ENVIRONMENT.md §8.4 patch to
verl's ToolAgentLoop — stock verl 0.9.0 renders a single image placeholder
for the whole list, and vLLM then dies with "Failed to apply prompt
replacement for mm_items['image'][1]". preflight.sh guards the patch.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

from agentic_tvg.constants import (
    CROP_MAX_PIXELS,
    CROP_MIN_PIXELS,
    CROP_NUM_FRAMES,
    MIN_CROP_SECONDS,
    TOOL_NAME,
)
from agentic_tvg.video_frames import sample_frames

logger = logging.getLogger(__name__)


def build_crop_video_schema(num_frames: int = CROP_NUM_FRAMES) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name=TOOL_NAME,
            description=(
                f"Inspect a time interval of the current video closely: returns {num_frames} "
                "higher-resolution frames sampled evenly from [start_time, end_time], "
                "each labeled with its timestamp in seconds."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "start_time": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Interval start in seconds, >= 0.",
                    ),
                    "end_time": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Interval end in seconds, > start_time, <= video duration.",
                    ),
                },
                required=["start_time", "end_time"],
            ),
        ),
    )


class CropVideoTool(BaseTool):
    """Stateful crop tool: one instance per trajectory, keyed by instance_id."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        self.num_frames = int(config.get("num_frames", CROP_NUM_FRAMES))
        self.max_pixels = int(config.get("max_pixels", CROP_MAX_PIXELS))
        self.min_pixels = int(config.get("min_pixels", CROP_MIN_PIXELS))
        self.min_crop_seconds = float(config.get("min_crop_seconds", MIN_CROP_SECONDS))
        self._instances: dict[str, dict[str, Any]] = {}
        super().__init__(config, tool_schema)  # calls get_openai_tool_schema() if schema is None

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        # Must not read self.tool_schema: BaseTool.__init__ calls this before it is set.
        return build_crop_video_schema(self.num_frames)

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        ck = kwargs.get("create_kwargs") or {}
        instance_id = instance_id or str(uuid4())
        self._instances[instance_id] = {
            "video_path": ck.get("video_path"),
            "duration": float(ck["duration"]) if ck.get("duration") is not None else None,
            "num_calls": 0,
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances.get(instance_id)
        if state is None or not state.get("video_path"):
            return ToolResponse(text="Error: no video is attached to this conversation."), 0.0, {}

        try:
            start = float(parameters["start_time"])
            end = float(parameters["end_time"])
        except (KeyError, TypeError, ValueError):
            return (
                ToolResponse(
                    text="Error: crop_video requires numeric start_time and end_time in seconds."
                ),
                0.0,
                {"error": "bad_args"},
            )

        duration = state["duration"]
        start, end, note = _normalize_window(start, end, duration, self.min_crop_seconds)
        if start is None:
            return ToolResponse(text=note), 0.0, {"error": "bad_window"}

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        try:
            frames, timestamps = await loop.run_in_executor(
                None,
                lambda: sample_frames(
                    state["video_path"], start, end, self.num_frames, self.max_pixels, self.min_pixels
                ),
            )
        except Exception as e:  # decode failures must not kill the rollout
            logger.warning("crop_video decode failed on %s [%s, %s]: %s", state["video_path"], start, end, e)
            return ToolResponse(text=f"Error: could not decode video interval [{start:.1f}, {end:.1f}]."), 0.0, {
                "error": "decode_failed"
            }
        decode_s = time.monotonic() - t0

        state["num_calls"] += 1
        text = crop_response_text(start, end, timestamps, note)
        metrics = {"decode_seconds": round(decode_s, 3), "window": [start, end], "num_frames": len(frames)}
        return ToolResponse(image=frames, text=text), 0.0, metrics

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)


def crop_response_text(start: float, end: float, timestamps: list[float], note: str = "") -> str:
    """The tool-response wording. Shared with data_prep/render_traces.py so SFT
    traces are re-rendered byte-identically to what the tool emits in RL."""
    return (
        f"The {len(timestamps)} frames above are sampled evenly from {start:.1f}s to {end:.1f}s"
        f"{note}. Their timestamps are: " + ", ".join(f"{t:.1f}s" for t in timestamps) + "."
    )


def _normalize_window(
    start: float, end: float, duration: Optional[float], min_seconds: float
) -> tuple[Optional[float], Optional[float], str]:
    """Clamp to [0, duration], expand too-narrow windows; (None, None, msg) if hopeless."""
    note = ""
    if duration is not None:
        start = min(max(start, 0.0), duration)
        end = min(max(end, 0.0), duration)
    else:
        start, end = max(start, 0.0), max(end, 0.0)

    if end <= start:
        return None, None, (
            f"Error: invalid interval [{start:.1f}, {end:.1f}] — end_time must be greater than "
            "start_time and inside the video."
        )

    if end - start < min_seconds:
        mid = (start + end) / 2
        start, end = mid - min_seconds / 2, mid + min_seconds / 2
        if start < 0:
            start, end = 0.0, min_seconds
        if duration is not None and end > duration:
            end = duration
            start = max(0.0, end - min_seconds)
        note = f" (requested window was narrower than {min_seconds:.0f}s and was expanded)"
    return start, end, note
