import numpy as np

from agentic_tvg.reward import PENALTY_BETA, compute_score, compute_score_penalty


def test_perfect_answer():
    r = compute_score(solution_str="<think>x</think><answer>[10, 20]</answer>", ground_truth=[10, 20])
    assert r["score"] == 1.5 and r["iou"] == 1.0 and r["format_score"] == 0.5


def test_no_answer():
    r = compute_score(solution_str="I don't know", ground_truth=[10, 20])
    assert r["score"] == 0.0 and r["answered"] == 0.0


def test_fallback_gets_iou_but_no_format_bonus():
    r = compute_score(solution_str="the segment is [10, 20]", ground_truth=[10.0, 20.0])
    assert r["iou"] == 1.0 and r["format_score"] == 0.0 and r["score"] == 1.0


def test_gt_normalization():
    for gt in ([10, 20], (10.0, 20.0), np.array([10.0, 20.0]), "[10, 20]"):
        r = compute_score(solution_str="<answer>[10, 20]</answer>", ground_truth=gt)
        assert r["iou"] == 1.0, gt


def test_tool_call_count():
    s = "<tool_call>{}</tool_call>x<tool_call>{}</tool_call><answer>[1, 2]</answer>"
    assert compute_score(solution_str=s, ground_truth=[1, 2])["num_tool_calls"] == 2.0


def test_penalty_kicks_in_only_when_inflated():
    gt = [40, 50]  # len 10
    tight = compute_score_penalty(
        solution_str="<answer>[40, 50]</answer>", ground_truth=gt, extra_info={"duration": 100}
    )
    assert tight["length_penalty"] == 0.0

    inflated_len = PENALTY_BETA * 10 + 30  # 30s beyond the tolerated 2x GT
    inflated = compute_score_penalty(
        solution_str=f"<answer>[10, {10 + inflated_len}]</answer>",
        ground_truth=gt,
        extra_info={"duration": 100},
    )
    assert inflated["length_penalty"] > 0
    vanilla = compute_score(
        solution_str=f"<answer>[10, {10 + inflated_len}]</answer>", ground_truth=gt
    )
    assert inflated["score"] < vanilla["score"]


def test_extra_keys_are_numeric():
    r = compute_score(solution_str="<answer>[1, 2]</answer>", ground_truth=[1, 2])
    assert all(isinstance(v, (int, float)) for v in r.values())


# --------------------------------------------------------------------------
# QA rewards. JUDGE_DISABLE=1 makes judge_answer return None so R_acc falls
# back to the deterministic alias matcher -- these tests are about the reward
# arithmetic, not the judge instrument (which is exercised by judge_audit2.py
# against real transcripts).
# --------------------------------------------------------------------------

import os

import pytest

from agentic_tvg.answer_match import expand_aliases
from agentic_tvg.reward import (
    TIME_WEIGHT,
    TIME_WEIGHT_V2,
    compute_score_qa,
    compute_score_qa2,
)

# The parquet carries GT already expanded (extract_rl.py freezes the alias
# list at build time), so the fallback matcher sees normalized aliases -- pass
# the same shape here rather than a raw GT string.
ALIASES = expand_aliases("a red flag")


@pytest.fixture(autouse=True)
def _judge_off(monkeypatch):
    monkeypatch.setenv("JUDGE_DISABLE", "1")


def _traj(*windows, answer="a red flag"):
    calls = "".join(
        '<tool_call>\n{"name": "crop_video", "arguments": '
        f'{{"start_time": {s}, "end_time": {e}}}}}\n</tool_call>'
        for s, e in windows
    )
    return f"<think>looking</think>{calls}<think>found it</think><answer>{answer}</answer>"


EXTRA = {"question": "what is waved?", "gt_text": "a red flag", "video_segment": [10.0, 20.0]}


def test_qa2_differs_from_qa_only_by_the_iou_weight():
    sol = _traj((10, 20))          # iou 1.0, format ok, alias hit
    r1 = compute_score_qa(solution_str=sol, ground_truth=ALIASES, extra_info=EXTRA)
    r2 = compute_score_qa2(solution_str=sol, ground_truth=ALIASES, extra_info=EXTRA)
    assert r1["acc"] == r2["acc"] == 1.0
    assert r1["evidence_iou"] == r2["evidence_iou"] == 1.0
    assert r1["score"] == 0.5 + 1.0 + TIME_WEIGHT * 1.0        # 2.0
    assert r2["score"] == 0.5 + 1.0 + TIME_WEIGHT_V2 * 1.0     # 2.5
    assert {k: v for k, v in r1.items() if k != "score"} == {k: v for k, v in r2.items() if k != "score"}


def test_qa2_keeps_the_format_bonus_not_the_penalty_flip():
    """GRPO2_PLAN §3b proposed 0/-0.5; kept as +0.5/0 (user call 2026-09-01).

    A malformed trajectory must score 0 on the format term, never negative --
    the reward range stays [0, 2.5], so no dashboard rescale.
    """
    ok = compute_score_qa2(solution_str=_traj((10, 20)), ground_truth=ALIASES, extra_info=EXTRA)
    bad = compute_score_qa2(solution_str="a red flag, no tags at all",
                            ground_truth=ALIASES, extra_info=EXTRA)
    assert ok["format_score"] == 0.5
    assert bad["format_score"] == 0.0
    assert bad["score"] >= 0.0


def test_qa2_has_no_multi_crop_shaping_term():
    """Dropped with its enabler (§3d). A productive second crop pays exactly
    what max-IoU already pays -- nothing extra for having retried."""
    one = compute_score_qa2(solution_str=_traj((10, 20)),
                            ground_truth=ALIASES, extra_info=EXTRA)
    # first crop misses entirely, second one is perfect: iou_best - iou_first = 1.0,
    # which the dropped shaping term would have paid +0.25 for.
    two = compute_score_qa2(solution_str=_traj((60, 70), (10, 20)),
                            ground_truth=ALIASES, extra_info=EXTRA)
    assert two["num_tool_calls"] == 2.0 and one["num_tool_calls"] == 1.0
    assert two["evidence_iou"] == one["evidence_iou"] == 1.0
    assert two["score"] == one["score"]


def test_qa2_iou_is_best_over_windows_not_last():
    r = compute_score_qa2(solution_str=_traj((10, 20), (60, 70)),
                          ground_truth=ALIASES, extra_info=EXTRA)
    assert r["evidence_iou"] == 1.0


def test_qa_scores_are_never_negative_and_numeric():
    for fn in (compute_score_qa, compute_score_qa2):
        r = fn(solution_str="", ground_truth=ALIASES, extra_info=EXTRA)
        assert r["score"] == 0.0 and r["answered"] == 0.0 and r["judge_used"] == 0.0
        assert all(isinstance(v, float) for v in r.values())
