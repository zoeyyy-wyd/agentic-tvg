"""Second-opinion calibration audit for the live judge instrument (v2).

Generalizes judge_audit.py (which hardcoded the 260->267 flip set): re-grades
EVERY row of the given rollout jsonl(s) with a chosen auditor model, using the
LIVE v2 rubric imported from agentic_tvg.judge_v2 (so the audit can never
drift from the instrument it audits; --rubric v1 keeps the frozen
pre-2026-09-01 prompt for historical comparison).

Three comparisons per run:
  - vs the live v2 verdicts (looked up in judge_cache_v2.jsonl by the same
    normalized key judge.py uses) -- the CALIBRATION number. GRPO2_PLAN §3e:
    a judge model up-tier is on the table only if this fails (< ~90%).
  - vs the acc recorded in the jsonl (whatever instrument scored the rollout).
  - verdict distributions + every disagreement, for hand-reading.

Usage:
  python judge_audit2.py results/val-rft/val_rollouts/0.jsonl                # opus, rubric v2
  python judge_audit2.py <rollout.jsonl> --model claude-sonnet-5 --rubric v1
  python judge_audit2.py <a.jsonl> <b.jsonl> --out audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic  # noqa: E402

# _load_dotenv runs at import; _key/_PROMPT keep this audit byte-aligned with
# the v2 instrument. Import from judge_v2, NOT judge: since the 2026-09-01
# split, judge.py is the restored v1 (haiku, one-word) instrument and importing
# _PROMPT from it would silently audit v1 while labelling the result v2.
from agentic_tvg.judge_v2 import _CACHE_PATH, _key, _PROMPT as PROMPT_V2  # noqa: E402
from agentic_tvg.answer_match import parse_answer_qa  # noqa: E402

# The pre-2026-09-01 one-word rubric, frozen verbatim for A/B history.
PROMPT_V1 = (
    "Question: {q}\n"
    "Reference answer: {gt}\n"
    "Candidate answer: {a}\n\n"
    "Grade the candidate against the reference (LongVT rubric):\n"
    "FULL - same answer, wording may differ.\n"
    "PARTIAL - correct but incomplete, or a correct core with wrong details.\n"
    "INCORRECT - wrong, contradictory, or lists multiple alternative answers "
    "/ hedges between options.\n\n"
    "Compare them briefly, then end with a line: VERDICT: FULL or "
    "VERDICT: PARTIAL or VERDICT: INCORRECT"
)

GRADE = {"FULL": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
_VERDICT_RE = re.compile(r"VERDICT:\s*(FULL|PARTIAL|INCORRECT)")
_Q_RE = re.compile(r'Question: "(.*?)"\s*\nAnswer the question', re.S)


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    for r in rows:
        m = _Q_RE.search(r.get("input", ""))
        r["_q"] = m.group(1).strip() if m else "?"
        r["_a"] = parse_answer_qa(r.get("output", "")).answer or ""
    return rows


def live_verdicts(rows: list[dict]) -> dict[str, float]:
    """key -> verdict from the live v2 cache; absent keys were never judged."""
    out: dict[str, float] = {}
    if _CACHE_PATH.exists():
        for line in _CACHE_PATH.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[rec["k"]] = float(rec["v"])
            except (json.JSONDecodeError, KeyError):
                continue
    return {(_key(r["_q"], r["gts"], r["_a"])): out[k]
            for r in rows
            if (k := _key(r["_q"], r["gts"], r["_a"])) in out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--model", default="claude-opus-5", help="auditor model (second opinion)")
    ap.add_argument("--rubric", choices=["v1", "v2"], default="v2")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, help="write per-row verdicts json here")
    args = ap.parse_args()

    rows = load_rows(args.jsonl)
    prompt = PROMPT_V2 if args.rubric == "v2" else PROMPT_V1
    client = anthropic.Anthropic(timeout=60)

    def rejudge(r: dict) -> tuple[float | None, str]:
        if not r["_a"]:
            return 0.0, "(no answer tag)"
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=args.model, max_tokens=1000,
                    messages=[{"role": "user",
                               "content": prompt.format(q=r["_q"], gt=r["gts"], a=r["_a"])}])
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                hits = _VERDICT_RE.findall(text.upper())
                return (GRADE[hits[-1]] if hits else None), text.strip()
            except Exception:
                if attempt == 2:
                    return None, "(api failure)"
        return None, ""

    with ThreadPoolExecutor(args.threads) as ex:
        verdicts = list(ex.map(rejudge, rows))

    live = live_verdicts(rows)
    recs = []
    for r, (v, text) in zip(rows, verdicts):
        recs.append({
            "q": r["_q"], "gt": r["gts"], "a": r["_a"],
            "recorded_acc": float(r["acc"]),
            "live_v2": live.get(_key(r["_q"], r["gts"], r["_a"])),
            "auditor": v, "auditor_text": text,
        })

    from collections import Counter
    ok = [x for x in recs if x["auditor"] is not None]
    print(f"auditor = {args.model} + rubric {args.rubric}, n = {len(ok)}/{len(recs)}")
    with_live = [x for x in ok if x["live_v2"] is not None]
    if with_live:
        agree = sum(1 for x in with_live if abs(x["auditor"] - x["live_v2"]) < 1e-6)
        print(f"CALIBRATION vs live v2 verdicts: {agree}/{len(with_live)} "
              f"({agree / len(with_live):.0%})   [<~90% -> consider judge up-tier, GRPO2_PLAN §3e]")
    else:
        print("CALIBRATION vs live v2: no cache overlap (run the live judge on these rows first)")
    agree_rec = sum(1 for x in ok if abs(x["auditor"] - x["recorded_acc"]) < 1e-6)
    print(f"vs recorded acc in jsonl:        {agree_rec}/{len(ok)} ({agree_rec / len(ok):.0%})")
    print(f"auditor mean acc {sum(x['auditor'] for x in ok) / len(ok):.4f}  "
          f"dist {Counter(x['auditor'] for x in ok)}")
    if with_live:
        print(f"live v2 mean acc {sum(x['live_v2'] for x in with_live) / len(with_live):.4f}  "
              f"dist {Counter(x['live_v2'] for x in with_live)}")

    print("\ndisagreements vs live v2:")
    for x in with_live:
        if abs(x["auditor"] - x["live_v2"]) > 1e-6:
            print(f"  live={x['live_v2']} auditor={x['auditor']}  gt={str(x['gt'])[:60]!r}")
            print(f"    a={x['a'][:90]!r}")
            print(f"    auditor: {' '.join(x['auditor_text'].splitlines())[:180]}")

    if args.out:
        args.out.write_text(json.dumps(recs, indent=1, ensure_ascii=False))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
