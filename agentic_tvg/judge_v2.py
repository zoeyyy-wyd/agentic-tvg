"""Answer-equivalence judge, INSTRUMENT V2: Anthropic API, sonnet, question-
anchored rubric, disk-cached.

Sibling module `judge.py` is instrument v1 (haiku, one-word rubric) and stays
the default; it produced every v1 number under results/. Selection is by env
var in reward.py (`JUDGE_V=2`), so which instrument a run used is recorded in
its environment rather than in a source diff. The two are NOT comparable and
keep separate caches -- never average or trend v1 and v2 numbers together.

R_acc, one instrument (README "Reward"; revised 2026-08-26, the free matcher
fast-path was removed -- it saved ~$1-3/run and created a matcher-vs-judge
grading seam; instrument v2 2026-09-01, see the block above JUDGE_MODEL):
EVERY parsed answer goes to this judge, graded FULL 1.0 / PARTIAL 0.5 /
INCORRECT 0, enumerations and hedges instructed INCORRECT.
answer_match.py survives as the deliberate-offline fallback and test scorer.

Determinism & audit: the Claude 5 API has no temperature control, so a single
call is a nondeterministic draw; determinism comes from the append-only JSONL
cache keyed by (question, gt, normalized answer) -- the FIRST verdict for a
triple is the verdict, ever after. Rows record model + rubric, and loading
skips rows from any other instrument, so a cache file cannot silently serve
another judge's verdicts. The cache file doubles as the audit trail for the
paper.

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
    python -m agentic_tvg.judge_v2 "What does he wave?" "A red flag." "a crimson banner"
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from agentic_tvg.answer_match import normalize

# Instrument v2 (2026-09-01). The v1 one-word haiku judge was audited against
# opus/sonnet second opinions on the 114-row val set (scratchpad audits, and
# 45 same-normalized-triple verdict flips in the v1 cache):
#   - haiku(one-word) vs opus: 82% agreement; 16 of 21 disagreements were
#     haiku's PARTIALs (11 deserved FULL, 5 deserved 0) -- 0.5 was a refuge
#     verdict, and 30% of all 31K training verdicts sat in it.
#   - haiku WITH reasoning: 75% vs opus -- the gap is model, not format.
#   - sonnet + this v2 rubric: PARTIAL bucket 47 -> 11 rows, decisive splits;
#     haiku under the same rubric still hedged (27 PARTIALs). Hence the
#     sonnet default. Judge noise feeds GRPO's group-relative advantage
#     directly, which is why this matters more than the ~$3/1K-call delta.
# v1 verdicts live in judge_cache.jsonl and are NOT comparable; v2 gets its
# own cache file. Historic val jsonls can be re-scored offline for ~cents.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5")
_CACHE_PATH = Path(os.environ.get("JUDGE_CACHE", "data/processed/judge_cache_v2.jsonl"))
_TIMEOUT_S = float(os.environ.get("JUDGE_TIMEOUT", "30"))
_RETRIES = 3

_SYSTEM = ("You grade video question answering. Compare briefly, then end with a line: "
           "VERDICT: FULL or VERDICT: PARTIAL or VERDICT: INCORRECT")
_PROMPT = (
    "Question: {q}\n"
    "Reference answer: {gt}\n"
    "Candidate answer: {a}\n\n"
    "Grade whether the candidate answers the QUESTION the same way the reference does.\n"
    "The reference often contains scene details beyond what the question asks; omitting\n"
    "those NEVER lowers the grade. Wording, casing, punctuation, and brevity never matter.\n\n"
    "FULL - the candidate's answer to the asked question matches the reference's.\n"
    "  Extra correct context or omitted unasked detail is still FULL.\n"
    "PARTIAL - the question asks for several things and the candidate gets some right\n"
    "  while omitting or missing others; or the right entity with a wrong detail that\n"
    "  the question explicitly asks about.\n"
    "INCORRECT - the answer to the asked question is wrong or contradicts the\n"
    "  reference; or it hedges between alternatives / lists several answers; or it\n"
    "  describes the scene without actually answering the question.\n\n"
    "Compare briefly, then end with a line: VERDICT: FULL or VERDICT: PARTIAL "
    "or VERDICT: INCORRECT"
)
_VERDICT_RE = re.compile(r"VERDICT:\s*(FULL|PARTIAL|INCORRECT)")
_GRADE = {"FULL": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
_RUBRIC_V = "v2"   # bump together with _PROMPT/_SYSTEM; gates cache-row loading

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
                    # _key doesn't encode the instrument, so the row must: a row
                    # from another model/rubric (v1 rows have no rubric field at
                    # all) never answers for this one, even if the files get
                    # mixed up or a JUDGE_MODEL override forgets JUDGE_CACHE.
                    if rec.get("model") != JUDGE_MODEL or rec.get("rubric") != _RUBRIC_V:
                        continue
                    _cache[rec["k"]] = float(rec["v"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return _cache


def _append(key: str, verdict: float, question: str, gt: str, answer: str) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_PATH.open("a") as f:
        f.write(json.dumps({"k": key, "v": verdict, "q": question, "gt": gt,
                            "a": answer, "model": JUDGE_MODEL, "rubric": _RUBRIC_V},
                           ensure_ascii=False) + "\n")


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
    last_failure = None
    for attempt in range(_RETRIES):
        try:
            resp = _client.messages.create(
                model=JUDGE_MODEL,
                # Brief comparison + the VERDICT line. A response that runs past the
                # budget gets cut BEFORE the VERDICT line and is unparseable -- and
                # the Claude 5 API samples nondeterministically (no temperature
                # control), so a retry is a fresh draw, with double the budget in
                # case the comparison is genuinely long. Without this, one verbose
                # borderline case among ~31K training verdicts stops the whole run.
                max_tokens=700 * (attempt + 1),
                system=_SYSTEM,
                messages=[{"role": "user", "content": _PROMPT.format(q=question, gt=gt_text, a=answer)}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            hits = _VERDICT_RE.findall(text.upper())
            if hits:
                verdict = _GRADE[hits[-1]]   # last line wins; reasoning may quote the labels
                break
            last_failure = f"no VERDICT line (stop_reason={getattr(resp, 'stop_reason', '?')})"
        except Exception as exc:
            if attempt == _RETRIES - 1:
                raise JudgeUnavailable(
                    f"{JUDGE_MODEL} failed {_RETRIES}x ({type(exc).__name__}: {exc})") from exc
            last_failure = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    if verdict is None:
        raise JudgeUnavailable(
            f"{JUDGE_MODEL} gave no parseable verdict in {_RETRIES} attempts ({last_failure})")
    with _lock:
        _load_cache()[key] = verdict
        _append(key, verdict, question, gt_text, answer)
    return verdict


if __name__ == "__main__":
    import sys
    q, gt, a = sys.argv[1], sys.argv[2], sys.argv[3]
    print({"enabled": enabled(), "model": JUDGE_MODEL, "verdict": judge_answer(q, gt, a)})
