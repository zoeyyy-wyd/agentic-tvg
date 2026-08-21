from agentic_tvg.span import parse_answer_span, temporal_iou


def test_basic_tag():
    p = parse_answer_span("<think>...</think><answer>[12.5, 28.0]</answer>")
    assert p.span == (12.5, 28.0) and p.format_ok and p.source == "tag"


def test_tag_variants():
    assert parse_answer_span("<answer> [ 3 , 9 ] </answer>").span == (3.0, 9.0)
    assert parse_answer_span("<answer>12.5s to 28s</answer>").span == (12.5, 28.0)
    assert parse_answer_span("<answer>10-20</answer>").span == (10.0, 20.0)
    assert parse_answer_span("<Answer>[1, 2]</Answer>").format_ok  # case-insensitive


def test_last_tag_wins():
    text = "<answer>[1, 2]</answer> wait, let me correct: <answer>[5, 9]</answer>"
    assert parse_answer_span(text).span == (5.0, 9.0)


def test_fallback_bracket_pair():
    p = parse_answer_span("I think the event happens at [14.0, 22.5] in the video.")
    assert p.span == (14.0, 22.5) and not p.format_ok and p.source == "fallback"


def test_no_answer():
    p = parse_answer_span("I cannot find the event.")
    assert p.span is None and p.source == "none"
    assert parse_answer_span("").span is None


def test_reversed_span_not_valid():
    p = parse_answer_span("<answer>[20, 10]</answer>")
    assert p.span == (20.0, 10.0) and p.format_ok and not p.valid


def test_tool_call_json_not_picked_up():
    # hermes tool-call args are dicts, not bare pairs — must not trigger fallback
    text = '<tool_call>{"name": "crop_video", "arguments": {"start_time": 10, "end_time": 20}}</tool_call>'
    assert parse_answer_span(text).span is None


def test_iou():
    assert temporal_iou((0, 10), (0, 10)) == 1.0
    assert temporal_iou((0, 10), (5, 15)) == 1 / 3
    assert temporal_iou((0, 5), (5, 10)) == 0.0
    assert temporal_iou((0, 5), (10, 20)) == 0.0
    assert temporal_iou(None, (0, 10)) == 0.0
    assert temporal_iou((5, 5), (0, 10)) == 0.0  # degenerate pred
    assert abs(temporal_iou((2, 8), (0, 10)) - 0.6) < 1e-9
