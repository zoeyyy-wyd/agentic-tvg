"""Custom SFT dataset for Qwen3-VL multi-turn tool traces.

Loaded by verl's SFT trainer via ``data.custom_cls`` (see run_sft_tvg.sh).
Fixes two Qwen3-VL gaps in verl 0.9.0's stock MultiTurnSFTDataset, both
verified empirically in this repo:

1. **Vision position ids**: the stock dataset hardcodes the *Qwen2-VL*
   ``get_rope_index``. Qwen3-VL renders videos as per-timestamp vision blocks
   (``<t> <vision_start>frame<vision_end> ...``), so the Qwen2 routine walks
   off the end of ``video_grid_thw`` (IndexError). verl ships the correct
   ``verl.models.transformers.qwen3_vl.get_rope_index`` for its RL path; we
   swap it in at module level here.

2. **Video timestamps**: the stock dataset pre-decodes videos to tensors and
   passes no ``video_metadata``, so the processor invents timestamps assuming
   fps=24 — fatally wrong labels for temporal grounding. We instead hand the
   processor the video *path* and let it decode (torchcodec) with real
   metadata. Frame count and pixel budget come from
   ``data.apply_chat_template_kwargs`` (note: Qwen3-VL's ``size.longest_edge``
   is a whole-video pixel budget, i.e. per-frame budget x num_frames).
"""

from __future__ import annotations

import copy

from omegaconf import OmegaConf

import verl.utils.dataset.multiturn_sft_dataset as _m
from verl.models.transformers.qwen3_vl import get_rope_index as _qwen3_vl_get_rope_index
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

# Fix 1: swap the rope-index routine used inside MultiTurnSFTDataset.__getitem__.
_m.get_rope_index = _qwen3_vl_get_rope_index

# Silence one exact transformers 5.x nag, nothing else: verl passes our
# apply_chat_template_kwargs loose (**kwargs) instead of wrapped in
# processor_kwargs, so processing_utils warns on EVERY call -- then adopts the
# values on the very next line (processing_utils.py, "processor_kwargs =
# processor_kwargs_from_kwargs"). Harmless but it prints tens of thousands of
# lines per run and buries the real signal.
import logging as _logging


class _DropProcessorKwargsNag(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        return "have to be in `processor_kwargs`" not in record.getMessage()


_logging.getLogger("transformers.processing_utils").addFilter(_DropProcessorKwargsNag())

# Placeholder the parent class does not recognize, so it passes through as text.
_SLOT = "<|tvg_video_slot|>"


class Qwen3VLMultiTurnSFTDataset(MultiTurnSFTDataset):
    """MultiTurnSFTDataset with path-based (metadata-preserving) video loading."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hydra hands us OmegaConf containers, which fail transformers 5.x
        # strict kwargs validation (DictConfig is not a dict) — plainify them.
        if OmegaConf.is_config(self.apply_chat_template_kwargs):
            self.apply_chat_template_kwargs = OmegaConf.to_container(
                self.apply_chat_template_kwargs, resolve=True
            )

    def _build_messages(self, example: dict):
        # Fix 2: hide the videos column and swap <video> for a sentinel before
        # the parent runs, so it neither pre-decodes nor chokes on the
        # placeholder; afterwards expand the sentinel into path entries.
        videos = list(example.get(self.video_key) or [])
        example = {k: v for k, v in example.items() if k != self.video_key}
        example[self.messages_key] = [
            {
                **msg,
                "content": msg["content"].replace("<video>", _SLOT)
                if isinstance(msg["content"], str)
                else copy.deepcopy(msg["content"]),
            }
            for msg in example[self.messages_key]
        ]

        messages = super()._build_messages(example)

        offset = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                if _SLOT in content:
                    raise ValueError("videos column missing for a <video> placeholder")
                continue
            new_content, offset = self._expand_slots(content, videos, offset)
            message["content"] = new_content
        if offset != len(videos):
            raise ValueError(f"{len(videos)} videos in row but {offset} <video> placeholders used")
        return messages

    @staticmethod
    def _expand_slots(content: list, videos: list, offset: int) -> tuple[list, int]:
        out = []
        for seg in content:
            text = seg.get("text") if isinstance(seg, dict) else None
            if text is not None and _SLOT in text:
                parts = text.split(_SLOT)
                for i, part in enumerate(parts):
                    if part:
                        out.append({"type": "text", "text": part})
                    if i < len(parts) - 1:
                        video = videos[offset]
                        path = video["video"] if isinstance(video, dict) else video
                        out.append({"type": "video", "video": str(path)})
                        offset += 1
            else:
                out.append(seg)
        return out, offset
