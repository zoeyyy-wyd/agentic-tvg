"""Prompt builders for the Step-0 probe modes and RL training.

Modes (plan §3): "direct" (no tool), "tool_optional", "tool_forced".
Training uses "tool_optional" by default; Step 0 measures all three.

The tool JSON schema itself is *not* embedded here — it is injected by the
chat template (``tools=...``) at rollout/serving time. The system prompt only
states the interaction policy and the frame/token budget contract.
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
    "You are a precise video temporal grounding assistant. Given a video and a query, "
    "you locate the single time interval in the video that best matches the query.\n"
    f"You initially see {GLOBAL_NUM_FRAMES} low-resolution frames sampled evenly from the whole video; "
    "each frame is preceded by its timestamp in seconds.\n"
    "Always reason step by step inside <think></think> tags before anything else. "
    "Give your final answer as the start and end time in seconds, formatted exactly as "
    f"{ANSWER_OPEN}[start_time, end_time]{ANSWER_CLOSE}, for example {ANSWER_OPEN}[12.5, 28.0]{ANSWER_CLOSE}. "
    "Output exactly one answer tag and nothing after it."
)

_TOOL_POLICY = (
    f"\nYou may call the tool {TOOL_NAME}(start_time, end_time) up to {MAX_TOOL_CALLS} times. "
    f"It returns {CROP_NUM_FRAMES} higher-resolution frames sampled evenly from that interval, "
    "each labeled with its timestamp. "
    "Recommended strategy: scan the global frames, propose a coarse candidate window, "
    f"call {TOOL_NAME} to inspect it closely, refine the boundaries, then answer."
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


def build_user_prompt(query: str, duration: float) -> str:
    """User turn for training data; ``<video>`` is the verl placeholder that
    RLHFDataset replaces with the actual video entry from the ``videos`` column."""
    return (
        f"<video>\nThe video is {duration:.1f} seconds long. "
        f'Query: "{query.strip()}"\n'
        "Find the time interval when this happens in the video."
    )
