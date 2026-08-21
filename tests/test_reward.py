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
