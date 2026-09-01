"""Full-set second-opinion audit: opus re-grades EVERY row of a rollout jsonl,
vs the haiku verdict recorded in it (acc field). Generalizes judge_audit.py.

Usage: python judge_full_audit.py <rollout.jsonl> [out.json]
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/yw4636/agentic-tvg")
import agentic_tvg.judge  # noqa: F401  loads .env -> ANTHROPIC_API_KEY
import anthropic

PROMPT = """Question: {q}
Reference answer: {gt}
Candidate answer: {a}

Grade the candidate against the reference (LongVT rubric):
FULL - same answer, wording may differ.
PARTIAL - correct but incomplete, or a correct core with wrong details.
INCORRECT - wrong, contradictory, or lists multiple alternative answers / hedges between options.

Compare them briefly, then end with a line: VERDICT: FULL or VERDICT: PARTIAL or VERDICT: INCORRECT"""

GRADE = {"FULL": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
for r in rows:
    m = re.search(r'Question: "(.*?)"\s*\nAnswer the question', r["input"], re.S)
    r["_q"] = m.group(1).strip() if m else "?"
    ans = re.findall(r"<answer>(.*?)</answer>", r["output"], re.S)
    r["_a"] = ans[-1].strip() if ans else ""

client = anthropic.Anthropic()

def rejudge(r):
    resp = client.messages.create(
        model="claude-opus-5", max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT.format(q=r["_q"], gt=r["gts"], a=r["_a"])}])
    text = next(b.text for b in resp.content if b.type == "text")
    m = re.findall(r"VERDICT:\s*(FULL|PARTIAL|INCORRECT)", text)
    return (GRADE[m[-1]] if m else None), text.strip()

with ThreadPoolExecutor(8) as ex:
    verdicts = list(ex.map(rejudge, rows))

recs, disagree = [], []
for r, (opus, text) in zip(rows, verdicts):
    haiku = float(r["acc"])
    recs.append({"q": r["_q"], "gt": r["gts"], "a": r["_a"],
                 "haiku": haiku, "opus": opus, "opus_text": text})
    if opus is not None and abs(opus - haiku) > 1e-6:
        disagree.append(recs[-1])

n = len(recs)
print(f"n={n}  agreement={n - len(disagree)}/{n} ({(n - len(disagree)) / n:.0%})")
up = sum(1 for d in disagree if d["opus"] > d["haiku"])
down = len(disagree) - up
print(f"opus grades HIGHER than haiku on {up}, LOWER on {down}")
mh = sum(x["haiku"] for x in recs) / n
mo = sum(x["opus"] for x in recs if x["opus"] is not None) / sum(1 for x in recs if x["opus"] is not None)
print(f"mean acc: haiku={mh:.4f}  opus={mo:.4f}  delta={mo - mh:+.4f}")
from collections import Counter
print("haiku dist:", Counter(x["haiku"] for x in recs))
print("opus dist: ", Counter(x["opus"] for x in recs))
print()
for d in disagree:
    print(f"haiku={d['haiku']} opus={d['opus']}  gt={d['gt'][:70]!r}")
    print(f"   ans={d['a'][:110]!r}")
    last = d["opus_text"].splitlines()
    print(f"   opus: {' '.join(last)[:220]}")
    print()

if len(sys.argv) > 2:
    json.dump(recs, open(sys.argv[2], "w"), indent=1)
    print("wrote", sys.argv[2])
