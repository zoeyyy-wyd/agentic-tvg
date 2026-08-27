"""Prompt builders for the agentic video QA task.

Modes: "direct" (no tool), "tool_optional" (training default), "tool_forced".
The tool JSON schema itself is *not* embedded here — it is injected by the
chat template (``tools=...``) at rollout/serving time. The system prompt only
states the interaction policy and the frame/token budget contract.

BYTE-STABILITY WARNING: rendered SFT data (data/processed/*.parquet) and the
RL rows both embed these exact strings. Any edit here desynchronizes SFT,
RL serving, and evaluation until the data is re-rendered — treat every string
below as frozen.
"""

from __future__ import annotations

from agentic_tvg.constants import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    CROP_NUM_FRAMES,
    GLOBAL_NUM_FRAMES,
    MAX_TOOL_CALLS,
    TOOL_NAME,
)

MODES = ("direct", "tool_optional", "tool_forced")

_BASE = (
    "You are a precise video question answering assistant. Given a video and a question, "
    "you answer from visual evidence in the video.\n"
    f"You initially see {GLOBAL_NUM_FRAMES} low-resolution frames sampled evenly from the whole video; "
    "each frame is preceded by its timestamp in seconds.\n"
    "Always reason step by step inside <think></think> tags before anything else. "
    "Give your final answer concisely — a short word or phrase, formatted exactly as "
    f"{ANSWER_OPEN}your answer{ANSWER_CLOSE}, for example {ANSWER_OPEN}A red flag.{ANSWER_CLOSE}. "
    "Output exactly one answer tag and nothing after it."
)

_TOOL_POLICY = (
    f"\nYou may call the tool {TOOL_NAME}(start_time, end_time) up to {MAX_TOOL_CALLS} times. "
    f"It returns {CROP_NUM_FRAMES} higher-resolution frames sampled evenly from that interval, "
    "each labeled with its timestamp. "
    "Recommended strategy: scan the global frames, locate when the queried moment occurs, "
    f"call {TOOL_NAME} to inspect that interval closely, verify the visual details, then answer."
)

_FORCED = (
    f"\nYou MUST call {TOOL_NAME} at least once to verify your candidate window "
    "before giving the final answer."
)


def build_system_prompt(mode: str = "tool_optional") -> str:
    if mode == "direct":
        return _BASE + "\nAnswer directly from the provided frames."
    if mode == "tool_optional":
        return _BASE + _TOOL_POLICY
    if mode == "tool_forced":
        return _BASE + _TOOL_POLICY + _FORCED
    raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")


def build_user_prompt(question: str, duration: float) -> str:
    """User turn; ``<video>`` is the verl placeholder that RLHFDataset /
    the SFT dataset replaces with the actual video entry."""
    return (
        f"<video>\nThe video is {duration:.1f} seconds long. "
        f'Question: "{question.strip()}"\n'
        "Answer the question based on the video."
    )
