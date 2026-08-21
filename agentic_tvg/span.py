"""Answer-span parsing and temporal IoU.

Pure stdlib so it is importable from any environment (reward sandbox, probe,
analysis notebooks) without pulling in torch/verl.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A non-negative number: "12", "12.5". Timestamps are never negative, and
# excluding a leading "-" lets "10-20" parse as a range instead of (10, -20).
_NUM = r"(\d+(?:\.\d+)?)"
# Range separators seen in model output: "[12, 20]", "12.0 - 20.0", "12 to 20",
# "12s ~ 20s", "12 and 20".
_SEP = r"\s*(?:s\b|sec\b|seconds\b)?\s*(?:,|~|-|–|—|to\b|and\b)\s*"
_RANGE_RE = re.compile(_NUM + _SEP + _NUM)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Fallback: a strict bracketed pair "[12.5, 20.0]" anywhere in the text.
_BRACKET_PAIR_RE = re.compile(r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]")


@dataclass
class ParsedSpan:
    """Result of parsing a predicted span from model output."""

    start: float | None
    end: float | None
    format_ok: bool  # True iff the span came from a well-formed <answer> tag
    source: str  # "tag" | "fallback" | "none"

    @property
    def span(self) -> tuple[float, float] | None:
        if self.start is None or self.end is None:
            return None
        return (self.start, self.end)

    @property
    def valid(self) -> bool:
        """A span usable for IoU: present and start < end."""
        return self.span is not None and self.start < self.end


def parse_answer_span(text: str) -> ParsedSpan:
    """Extract the predicted [start, end] from model output.

    Primary path: the last non-empty ``<answer>...</answer>`` tag (the final
    answer in a multi-turn trace), taking the first number range inside it.
    Fallback path: the last bracketed pair ``[a, b]`` anywhere in the text —
    the model answered but broke the tag format, so ``format_ok`` stays False.
    """
    if not text:
        return ParsedSpan(None, None, False, "none")

    for tag_content in reversed(_ANSWER_TAG_RE.findall(text)):
        m = _RANGE_RE.search(tag_content)
        if m:
            return ParsedSpan(float(m.group(1)), float(m.group(2)), True, "tag")

    pairs = _BRACKET_PAIR_RE.findall(text)
    if pairs:
        a, b = pairs[-1]
        return ParsedSpan(float(a), float(b), False, "fallback")

    return ParsedSpan(None, None, False, "none")


def temporal_iou(pred: tuple[float, float] | None, gt: tuple[float, float] | None) -> float:
    """Standard 1-D IoU between two [start, end] intervals.

    Returns 0.0 for a missing or degenerate (start >= end) prediction — a
    reversed span is treated as wrong, not silently repaired, so degenerate
    output stays visible in the metrics.
    """
    if pred is None or gt is None:
        return 0.0
    ps, pe = pred
    gs, ge = gt
    if pe <= ps or ge <= gs:
        return 0.0
    inter = min(pe, ge) - max(ps, gs)
    if inter <= 0:
        return 0.0
    union = (pe - ps) + (ge - gs) - inter
    return inter / union
