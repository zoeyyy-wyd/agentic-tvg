"""verl custom reward functions for agentic TVG (plan §5 Stage 2).

Loaded by verl via ``reward.custom_reward_function.path=<this file>`` with
``name=compute_score`` (vanilla) or ``name=compute_score_penalty`` (the
penalty-aware ablation from plan §6.2). Signature and dict-return contract
follow verl 0.9.0's NaiveRewardManager.

``solution_str`` is the decoded multi-turn response, including tool-call and
tool-response text; the answer parser takes the *last* ``<answer>`` tag.
"""

from __future__ import annotations

import json
from typing import Any

from agentic_tvg.span import parse_answer_span, temporal_iou

FORMAT_BONUS = 0.5  # plan §5: r = r_fmt + IoU

# Penalty-aware variant (plan §6.2): discourage span inflation. A prediction
# longer than PENALTY_BETA x the GT span is taxed in proportion to the excess
# length relative to the video duration, capped at PENALTY_LAMBDA.
PENALTY_BETA = 2.0
PENALTY_LAMBDA = 0.5


def _normalize_gt(ground_truth: Any) -> tuple[float, float] | None:
    """Accept [s, e] as list/tuple/ndarray/JSON string and return a tuple."""
    gt = ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (json.JSONDecodeError, ValueError):
            parsed = parse_answer_span(f"<answer>{gt}</answer>")
            return parsed.span
    try:
        seq = list(gt)
    except TypeError:
        return None
    if len(seq) != 2:
        return None
    try:
        return (float(seq[0]), float(seq[1]))
    except (TypeError, ValueError):
        return None


def _base_score(solution_str: str, ground_truth: Any) -> dict:
    gt = _normalize_gt(ground_truth)
    parsed = parse_answer_span(solution_str or "")
    fmt = FORMAT_BONUS if parsed.format_ok else 0.0
    iou = temporal_iou(parsed.span if parsed.valid else None, gt)
    return {
        "score": fmt + iou,
        "iou": iou,
        "format_score": fmt,
        "answered": 1.0 if parsed.span is not None else 0.0,
        "pred_start": parsed.start if parsed.start is not None else -1.0,
        "pred_end": parsed.end if parsed.end is not None else -1.0,
        "num_tool_calls": float((solution_str or "").count("<tool_call>")),
        "_gt": gt,  # stripped before returning
        "_parsed": parsed,
    }


def _finalize(result: dict) -> dict:
    result.pop("_gt", None)
    result.pop("_parsed", None)
    return result


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Vanilla reward: format bonus + temporal IoU."""
    return _finalize(_base_score(solution_str, ground_truth))


def compute_score_penalty(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Penalty-aware reward: format bonus + IoU - span-inflation penalty."""
    result = _base_score(solution_str, ground_truth)
    gt, parsed = result["_gt"], result["_parsed"]

    penalty = 0.0
    if gt is not None and parsed.valid:
        gt_len = gt[1] - gt[0]
        pred_len = parsed.end - parsed.start
        duration = None
        if extra_info is not None:
            duration = extra_info.get("duration")
        norm = float(duration) if duration else max(gt[1], parsed.end)
        if norm > 0 and gt_len > 0:
            excess = max(0.0, pred_len - PENALTY_BETA * gt_len) / norm
            penalty = PENALTY_LAMBDA * min(1.0, excess)

    result["score"] -= penalty
    result["length_penalty"] = penalty
    return _finalize(result)


# --------------------------------------------------------------------------
# QA reward (PLAN.md 5): R = 0.5*format + R_acc + TIME_WEIGHT*IoU(crop, evidence)
# --------------------------------------------------------------------------

import re as _re

from agentic_tvg.judge import judge_answer
from agentic_tvg.answer_match import answer_matches, expand_aliases, parse_answer_qa

TIME_WEIGHT = 0.5  # lambda on the evidence-IoU term; 0 disables it (cut ablation)

_TOOL_CALL_RE = _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL)


def _crop_windows(solution_str: str) -> list[tuple[float, float]]:
    """Every crop_video window the policy called during the rollout."""
    out = []
    for blob in _TOOL_CALL_RE.findall(solution_str or ""):
        try:
            args = json.loads(blob).get("arguments", {})
            out.append((float(args["start_time"]), float(args["end_time"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def _aliases_of(ground_truth: Any) -> list[str]:
    """Accept a pre-expanded alias list (the parquet case) or a raw GT string."""
    if isinstance(ground_truth, str):
        return expand_aliases(ground_truth)
    try:
        return [str(a) for a in list(ground_truth)]
    except TypeError:
        return []


def compute_score_qa(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Verifiable QA reward: format bonus + alias match + evidence IoU.

    - R_acc: word-level alias containment with the anti-enumeration length cap
      (answer_match.py). ground_truth carries the frozen alias list.
    - R_time: best IoU between any crop_video window the policy actually
      called and extra_info["video_segment"]. No tool call -> 0, so evidence
      use is rewarded directly.
    """
    parsed = parse_answer_qa(solution_str or "")
    fmt = FORMAT_BONUS if parsed.format_ok else 0.0

    # R_acc (paper-faithful, revised 2026-08-26): EVERY parsed answer goes to
    # the cached temp-0 judge -- LongVT rubric {FULL 1.0, PARTIAL 0.5,
    # INCORRECT 0}. One instrument, no matcher/judge seam.
    #
    # The alias fallback below now fires ONLY when the judge is deliberately
    # off (no ANTHROPIC_API_KEY, or JUDGE_DISABLE=1), which judge.py announces
    # once on stdout. An enabled judge that fails raises JudgeUnavailable and
    # stops the run instead: swapping a binary matcher in for the {0, 0.5, 1}
    # rubric mid-training silently changes what we are optimising.
    acc, judge_used = 0.0, 0.0
    if parsed.answer:
        ei = extra_info or {}
        gt_text = str(ei.get("gt_text", "") or ground_truth or "")
        verdict = judge_answer(str(ei.get("question", "")), gt_text, parsed.answer)
        if verdict is not None:
            judge_used = 1.0
            acc = float(verdict)
        else:
            acc = 1.0 if answer_matches(parsed.answer, _aliases_of(ground_truth)) else 0.0

    seg = None
    if extra_info is not None:
        seg = _normalize_gt(extra_info.get("video_segment"))
    windows = _crop_windows(solution_str)
    evidence_iou = max((temporal_iou(w, seg) for w in windows), default=0.0) if seg else 0.0

    return {
        "score": fmt + acc + TIME_WEIGHT * evidence_iou,
        "acc": acc,
        "format_score": fmt,
        "evidence_iou": evidence_iou,
        "answered": 1.0 if parsed.answer is not None else 0.0,
        "num_tool_calls": float(len(windows)),
        "judge_used": judge_used,
    }
