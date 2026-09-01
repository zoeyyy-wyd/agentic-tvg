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
# QA reward (README "Reward"): R = 0.5*format + judge R_acc + TIME_WEIGHT*IoU(crop, evidence)
# --------------------------------------------------------------------------

import os as _os
import re as _re

# R_acc instrument selector (split 2026-09-01; **default flipped to v2
# 2026-09-01** when the recipe moved to round 2). judge.py is the v1
# instrument (haiku, one-word rubric) that produced every v1 number in
# results/; judge_v2.py is the sonnet + question-anchored-rubric instrument
# and is now the default. Select with the env var, never by editing this
# import: the instrument a run used is then recorded in its environment and
# hydra config, not in a diff nobody re-reads. v1 and v2 verdicts are NOT
# comparable (separate caches) -- pass JUDGE_V=1 to reproduce a v1 number.
#
# The default used to be "1" while every reported v2 number came from OFFLINE
# re-grading (judge_audit2.py over dumped rollouts), so a training run that
# forgot to export JUDGE_V=2 would silently optimise against v1 for its whole
# horizon. run_grpo.sh now exports and prints the choice for the same reason.
JUDGE_V = _os.environ.get("JUDGE_V", "2")
if JUDGE_V == "2":
    from agentic_tvg.judge_v2 import judge_answer
else:
    from agentic_tvg.judge import judge_answer
from agentic_tvg.answer_match import answer_matches, expand_aliases, parse_answer_qa

TIME_WEIGHT = 0.5  # lambda on the evidence-IoU term; 0 disables it (cut ablation)

# Round 2 (GRPO2_PLAN §3b): iou is the plateaued target, so its term doubles.
# Calibration measured offline on round 1's own 34,048 rollouts (2,128 complete
# 16-groups, results/grpo-vanilla/rollouts_grpo267): raising 0.5 -> 1.0 leaves
# the within-group advantage ranking at Spearman 0.992 and changes the winning
# trajectory in only 3.1% of groups (1.3% over the plateau, step >= 180). So
# this is NOT the lever that moves iou: inside an acc-tied subgroup iou already
# decides the ranking at ANY positive weight (verified: 0/1860 tie-break flips
# across w in 0.5..5.0). What the weight actually buys is cross-tier authority
# -- how often better grounding may outrank a better answer. At 1.0 the iou
# spread (std 0.171) reaches 64% of acc's (0.268), i.e. a 0.5 iou gap can
# overturn one acc tier; at 0.5 it cannot. Quote that, not "pushes grounding".
TIME_WEIGHT_V2 = 1.0

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


def _qa_score(solution_str: str, ground_truth: Any, extra_info: dict | None,
              time_weight: float) -> dict:
    """Shared body of compute_score_qa / compute_score_qa2.

    The two public entry points differ ONLY in ``time_weight``; everything
    else -- format bonus, judge instrument, window parsing -- is common, so
    the round-1 function keeps returning exactly what it always did.
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
        "score": fmt + acc + time_weight * evidence_iou,
        "acc": acc,
        "format_score": fmt,
        "evidence_iou": evidence_iou,
        "answered": 1.0 if parsed.answer is not None else 0.0,
        "num_tool_calls": float(len(windows)),
        "judge_used": judge_used,
    }


def compute_score_qa(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Round-1 QA reward: 0.5*format + judge R_acc + 0.5*evidence-IoU.

    - R_acc: the cached judge's {0, 0.5, 1} verdict, with word-level alias
      containment (answer_match.py) as the deliberate-offline fallback only.
    - R_time: best IoU between any crop_video window the policy actually
      called and extra_info["video_segment"]. No tool call -> 0, so evidence
      use is rewarded directly.

    Frozen for comparability with `grpo-vanilla`. Note the instrument behind
    R_acc is NOT frozen with it -- JUDGE_V selects that, and its default is
    now v2, so re-running this function does not reproduce a v1 number
    unless JUDGE_V=1 is also set.
    """
    return _qa_score(solution_str, ground_truth, extra_info, TIME_WEIGHT)


def compute_score_qa2(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Round-2 QA reward (GRPO2_PLAN §3b as revised 2026-09-01).

    Exactly compute_score_qa with TIME_WEIGHT_V2 = 1.0 on the IoU term.
    Two things the drafted §3b asked for are deliberately NOT here:

    - **the format flip** (`+0.5 if ok` -> `0 / -0.5`). Dropped on the user's
      call 2026-09-01: it is a uniform -0.5 shift on every trajectory and
      GRPO's group-normalized advantage is exactly invariant to a constant
      shift, so it buys no gradient and costs a dashboard rescale plus a code
      path that diverges from `_base_score`. The bonus form stays.
    - **the multi-crop shaping term** `+0.25*max(0, iou_best - iou_first)`.
      Dropped with its enabler (§3d data injection was abandoned 2026-09-01
      after the reflection traces were measured against our frame budget).
      Round 1 sampled 22 multi-crop trajectories in 34,048, so the term would
      be identically 0; §3 itself notes the shaping and the enabler only work
      together.

    So the round-2 gradient changes are the judge rubric (v2, now the default
    instrument) and this weight -- and the offline calibration in the
    TIME_WEIGHT_V2 comment says the judge is the larger of the two by ~7x.
    """
    return _qa_score(solution_str, ground_truth, extra_info, TIME_WEIGHT_V2)
