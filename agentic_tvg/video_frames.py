"""PyAV-based interval frame sampling.

Shared by the crop_video tool (verl rollout) and the Step-0 probe so both see
byte-identical frames for the same request. Frames are resized here, to
dimensions aligned to Qwen3-VL's 32x32 token block, so the number of vision
tokens per frame is decided by our budget (constants.py) rather than by the
processor's own defaults.
"""

from __future__ import annotations

import math

import av
import numpy as np
from PIL import Image

# Qwen3-VL: patch 16 x spatial merge 2 -> one token per 32x32 pixel block.
_ALIGN = 32


def get_video_duration(path: str) -> float:
    """Video duration in seconds (container metadata, no decoding)."""
    with av.open(path) as container:
        if container.duration is not None:
            return container.duration / av.time_base
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
    raise ValueError(f"cannot determine duration of {path}")


def _fit_size(width: int, height: int, max_pixels: int, min_pixels: int) -> tuple[int, int]:
    """Target (w, h): aspect-preserving, inside [min_pixels, max_pixels], 32-aligned."""
    pixels = width * height
    scale = 1.0
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    w = max(_ALIGN, int(round(width * scale / _ALIGN)) * _ALIGN)
    h = max(_ALIGN, int(round(height * scale / _ALIGN)) * _ALIGN)
    # Rounding up on both axes can overshoot max_pixels; shrink the longer side.
    while w * h > max_pixels and max(w, h) > _ALIGN:
        if w >= h:
            w -= _ALIGN
        else:
            h -= _ALIGN
    return w, h


def sample_frames(
    path: str,
    start: float,
    end: float,
    num_frames: int,
    max_pixels: int,
    min_pixels: int,
) -> tuple[list[Image.Image], list[float]]:
    """Decode ``num_frames`` frames evenly spanning [start, end] seconds.

    Returns (frames, timestamps) where timestamps are the *actual* decoded
    frame times (seconds), which the caller should surface to the model.
    [start, end] is assumed already clamped/validated by the caller.
    Frames past EOF repeat the last decoded frame so the count is stable.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if end <= start:
        raise ValueError(f"need end > start, got [{start}, {end}]")

    targets = np.linspace(start, end, num_frames)

    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        # Seek to the nearest keyframe at/before `start` (offset in av.time_base).
        container.seek(max(0, int(start * av.time_base)), backward=True)

        size: tuple[int, int] | None = None
        picked: list[tuple[float, Image.Image]] = []
        prev: tuple[float, av.VideoFrame] | None = None
        ti = 0

        def _emit(t: float, frame: av.VideoFrame) -> None:
            nonlocal size
            img = frame.to_image()
            if size is None:
                size = _fit_size(img.width, img.height, max_pixels, min_pixels)
            picked.append((t, img.resize(size, Image.LANCZOS)))

        for frame in container.decode(stream):
            t = frame.time
            if t is None:
                continue
            while ti < len(targets) and t >= targets[ti]:
                # Pick whichever of (previous, current) frame is closer.
                if prev is not None and abs(prev[0] - targets[ti]) <= abs(t - targets[ti]):
                    _emit(prev[0], prev[1])
                else:
                    _emit(t, frame)
                ti += 1
            if ti >= len(targets):
                break
            prev = (t, frame)

        # Targets beyond the last decoded frame (EOF): repeat the final frame.
        if ti < len(targets):
            if prev is not None:
                _emit(prev[0], prev[1])
                ti += 1
            while picked and ti < len(targets):
                picked.append(picked[-1])
                ti += 1

    if not picked:
        raise ValueError(f"decoded no frames from {path} in [{start}, {end}]")
    frames = [img for _, img in picked]
    timestamps = [round(t, 2) for t, _ in picked]
    return frames, timestamps
