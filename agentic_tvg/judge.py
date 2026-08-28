"""Answer-equivalence judge: Anthropic API, LongVT's rubric, temp 0, disk-cached.

R_acc, one instrument (README "Reward"; revised 2026-08-26, the free matcher
fast-path was removed -- it saved ~$1-3/run and created a matcher-vs-judge
grading seam): EVERY parsed answer goes to this judge, graded FULL 1.0 /
PARTIAL 0.5 / INCORRECT 0, enumerations and hedges instructed INCORRECT.
answer_match.py survives as the deliberate-offline fallback and test scorer.

Determinism & audit: temperature 0 plus an append-only JSONL cache keyed by
(question, gt, normalized answer) -- one verdict per unique triple, ever.
The cache file doubles as the audit trail for the paper.

Failure semantics (hardened 2026-08-28, after a mid-run credit outage):
  - deliberately off (no ANTHROPIC_API_KEY, or JUDGE_DISABLE=1) -> returns
    None, announced once on stdout; caller falls back to alias matching.
    Offline smokes/tests only -- not comparable to judged runs.
  - enabled but failing (API error after retries, unparseable verdict,
    missing GT) -> raises JudgeUnavailable and STOPS the run. A silent
    fallback would swap scoring instruments mid-training; the cache makes
    the resume free. Credits: console.anthropic.com balance, NOT claude.ai
    usage credits -- different pool, identical error text.

Manual smoke once the key is exported:
    python -m agentic_tvg.judge "What does he wave?" "A red flag." "a crimson banner"
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from agentic_tvg.answer_match import normalize

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")
_CACHE_PATH = Path(os.environ.get("JUDGE_CACHE", "data/processed/judge_cache.jsonl"))
_TIMEOUT_S = float(os.environ.get("JUDGE_TIMEOUT", "20"))
_RETRIES = 3

_SYSTEM = "You grade video question answering. Reply with exactly one word: FULL, PARTIAL, or INCORRECT."
_PROMPT = (
    "Question: {q}\n"
    "Reference answer: {gt}\n"
    "Candidate answer: {a}\n\n"
    "Grade the candidate against the reference (LongVT rubric):\n"
    "FULL - same answer, wording may differ.\n"
    "PARTIAL - correct but incomplete, or a correct core with wrong details.\n"
    "INCORRECT - wrong, contradictory, or lists multiple alternative answers "
    "/ hedges between options.\n"
    "Reply with exactly one word."
)

def _load_dotenv() -> None:
    """Repo-root .env (gitignored, chmod 600). Shell-exported values win."""
    envf = Path(__file__).resolve().parents[1] / ".env"
    if not envf.exists():
        return
    for line in envf.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.removeprefix("export ").partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


_load_dotenv()
# re-read after .env so the module-level defaults see it
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", JUDGE_MODEL)

_lock = threading.Lock()
_cache: dict[str, float] | None = None
_client = None


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY")) and os.environ.get("JUDGE_DISABLE") != "1"


def _key(question: str, gt: str, answer: str) -> str:
    blob = "\x1f".join((question.strip(), gt.strip(), normalize(answer)))
    return hashlib.sha1(blob.encode()).hexdigest()


def _load_cache() -> dict[str, float]:
    global _cache
    if _cache is None:
        _cache = {}
        if _CACHE_PATH.exists():
            for line in _CACHE_PATH.read_text().splitlines():
                try:
                    rec = json.loads(line)
                    _cache[rec["k"]] = float(rec["v"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return _cache


def _append(key: str, verdict: float, question: str, gt: str, answer: str) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("a") as f:
        f.write(json.dumps({"k": key, "v": verdict, "q": question, "gt": gt,
                            "a": answer, "model": JUDGE_MODEL}, ensure_ascii=False) + "\n")


class JudgeUnavailable(RuntimeError):
    """The judge was asked and did not answer.

    Raised, never swallowed. R_acc scored by the alias matcher is a *different
    instrument* from the LongVT rubric -- binary instead of {0, 0.5, 1}, and
    stricter -- so a silent fallback rewrites the reward function mid-run and
    leaves no trace. Over a 52h GRPO run that corrupts the training signal in a
    way no metric would reveal after the fact. verl's NaiveRewardManager only
    catches TimeoutError (naive.py:137), so this propagates and stops the run;
    the judge cache is append-only, so resuming re-pays for nothing.
    """


_announced = False


def _announce_once() -> None:
    """The deliberate offline path still has to say so, exactly once."""
    global _announced
    if not _announced:
        _announced = True
        print("[judge] ANTHROPIC_API_KEY unset or JUDGE_DISABLE=1 -- R_acc falls back to "
              "strict alias matching (binary, no PARTIAL). Not comparable to judged runs.",
              flush=True)


def judge_answer(question: str, gt_text: str, answer: str) -> float | None:
    """1.0 / 0.5 / 0.0 = FULL / PARTIAL / INCORRECT.

    Returns None only when the judge is deliberately off (no key / JUDGE_DISABLE),
    which is announced once. Every other failure raises JudgeUnavailable.
    """
    if not enabled():
        _announce_once()
        return None
    if not answer:
        return None  # nothing was parsed out; the caller scores this as 0
    if not gt_text:
        raise JudgeUnavailable("row has no ground-truth text to grade against")
    key = _key(question, gt_text, answer)
    with _lock:
        cache = _load_cache()
        if key in cache:
            return cache[key]

    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(timeout=_TIMEOUT_S)

    verdict = None
    for attempt in range(_RETRIES):
        try:
            resp = _client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=4,
                extra_body={"temperature": 0},   # SDK 1.0 removed the kwarg; the API still takes it
                system=_SYSTEM,
                messages=[{"role": "user", "content": _PROMPT.format(q=question, gt=gt_text, a=answer)}],
            )
            text = resp.content[0].text.strip().upper()
            if text.startswith("FULL"):
                verdict = 1.0
            elif text.startswith("PARTIAL"):
                verdict = 0.5
            elif text.startswith("INCORRECT"):
                verdict = 0.0
            break
        except Exception as exc:
            if attempt == _RETRIES - 1:
                raise JudgeUnavailable(
                    f"{JUDGE_MODEL} failed {_RETRIES}x ({type(exc).__name__}: {exc})") from exc
            time.sleep(1.5 * (attempt + 1))
    if verdict is None:
        raise JudgeUnavailable(f"{JUDGE_MODEL} returned an unparseable verdict")
    with _lock:
        _load_cache()[key] = verdict
        _append(key, verdict, question, gt_text, answer)
    return verdict


if __name__ == "__main__":
    import sys
    q, gt, a = sys.argv[1], sys.argv[2], sys.argv[3]
    print({"enabled": enabled(), "model": JUDGE_MODEL, "verdict": judge_answer(q, gt, a)})
