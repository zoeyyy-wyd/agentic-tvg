"""Verifiable answer checking for the QA reward (PLAN.md 5).

Three pieces, all deterministic:
- normalize():      lowercase, strip punctuation and articles
- expand_aliases(): GT string -> frozen alias set (rule-based: parenthetical
                    variants like "a dark gray stone (rock)", number words).
                    Run once at data-prep time; LLM enrichment appends to the
                    same list later. Aliases are stored *normalized*.
- answer_matches(): containment at word granularity (alias tokens must appear
                    as a contiguous sublist -- "red" never matches "bored"),
                    plus the anti-enumeration length cap: an answer longer
                    than len(shortest alias) + LENGTH_SLACK words scores 0,
                    so "red orange yellow blue green" cannot farm color GTs.

Error asymmetry rationale (PLAN.md 5): a false negative zeroes a GRPO group
-> zero variance -> the sample is skipped (safe). Strictness is the point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_PAREN = re.compile(r"\(([^)]*)\)")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

LENGTH_SLACK = 4  # answer may exceed the shortest alias by at most this many words

_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}
_NUM_DIGITS = {v: k for k, v in _NUM_WORDS.items()}


def normalize(s: str) -> str:
    s = _NON_ALNUM.sub(" ", s.lower())
    return " ".join(_ARTICLES.sub(" ", s).split())


def _number_variants(norm: str) -> set[str]:
    toks = norm.split()
    out = set()
    for table in (_NUM_WORDS, _NUM_DIGITS):
        if any(t in table for t in toks):
            out.add(" ".join(table.get(t, t) for t in toks))
    return out


def expand_aliases(ground_truth: str) -> list[str]:
    """GT text -> normalized alias list (order: most specific first)."""
    variants: list[str] = []
    base = ground_truth.strip()
    # parenthetical alternatives: keep the full text, the text with parens
    # removed, and each parenthetical content on its own
    stripped = _PAREN.sub(" ", base)
    candidates = [base, stripped] + _PAREN.findall(base)
    for c in candidates:
        n = normalize(c)
        if n and n not in variants:
            variants.append(n)
    for n in list(variants):
        for v in _number_variants(n):
            if v not in variants:
                variants.append(v)
    return variants


def _contains(answer_toks: list[str], alias_toks: list[str]) -> bool:
    n, m = len(answer_toks), len(alias_toks)
    if m == 0 or m > n:
        return False
    return any(answer_toks[i:i + m] == alias_toks for i in range(n - m + 1))


def containment_matches(answer: str, aliases: list[str]) -> bool:
    """Word-level containment of any alias (no length cap)."""
    ans = normalize(answer).split()
    if not ans or not aliases:
        return False
    return any(_contains(ans, a.split()) for a in aliases if a)


def within_length_cap(answer: str, aliases: list[str]) -> bool:
    """Anti-enumeration gate: answer may exceed the shortest alias by at most
    LENGTH_SLACK words. Since 2026-08-26 this is a *router*, not a scorer:
    over-cap answers are not auto-zeroed but must face the tier-2 judge."""
    ans = normalize(answer).split()
    toks = [a.split() for a in aliases if a]
    if not ans or not toks:
        return False
    return len(ans) <= min(len(t) for t in toks) + LENGTH_SLACK


def answer_matches(answer: str, aliases: list[str]) -> bool:
    """Tier-1 fast path: containment AND within the cap."""
    return within_length_cap(answer, aliases) and containment_matches(answer, aliases)


@dataclass
class ParsedAnswer:
    answer: str | None      # last <answer> content, stripped
    format_ok: bool         # exactly one answer tag, nothing after it, has <think>


def parse_answer_qa(text: str) -> ParsedAnswer:
    tags = _ANSWER.findall(text or "")
    answer = tags[-1].strip() if tags else None
    format_ok = (
        len(tags) == 1
        and (text or "").rstrip().endswith("</answer>")
        and bool(_THINK.search(text or ""))
    )
    return ParsedAnswer(answer=answer, format_ok=format_ok)
