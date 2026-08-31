"""Second-opinion audit of the haiku judge on the 260->267 val flips.

Re-grades both answers of every acc-flipped sample with claude-opus-5 using
the same LongVT rubric, but with reasoning allowed before the verdict.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import agentic_tvg.judge  # noqa: F401  (imports repo .env -> ANTHROPIC_API_KEY)
import anthropic

ROLL = "/home/yw4636/agentic-tvg/results/grpo-vanilla/val_rollouts"

PROMPT = """Question: {q}
Reference answer: {gt}
Candidate answer: {a}

Grade the candidate against the reference (LongVT rubric):
FULL - same answer, wording may differ.
PARTIAL - correct but incomplete, or a correct core with wrong details.
INCORRECT - wrong, contradictory, or lists multiple alternative answers / hedges between options.

Compare them briefly, then end with a line: VERDICT: FULL or VERDICT: PARTIAL or VERDICT: INCORRECT"""

GRADE = {"FULL": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def load(name):
    with open(f"{ROLL}/{name}.jsonl") as f:
        return [json.loads(l) for l in f]


def question_of(rec):
    m = re.search(r'Question: "(.*?)"\n', rec["input"], re.S)
    return m.group(1) if m else rec["input"][-400:]


def answer_of(rec):
    m = re.findall(r"<answer>(.*?)</answer>", rec["output"], re.S)
    return m[-1].strip() if m else ""


client = anthropic.Anthropic()


def rejudge(q, gt, a):
    resp = client.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": PROMPT.format(q=q, gt=gt, a=a)}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    m = re.findall(r"VERDICT:\s*(FULL|PARTIAL|INCORRECT)", text)
    return (GRADE[m[-1]] if m else None), text.strip().splitlines()


b, c = load("260"), load("267")
flips = [i for i in range(len(b)) if b[i]["acc"] != c[i]["acc"]]
jobs = []
for i in flips:
    q, gt = question_of(b[i]), b[i]["gts"]
    jobs.append((i, "260", q, gt, answer_of(b[i]), b[i]["acc"]))
    jobs.append((i, "267", q, gt, answer_of(c[i]), c[i]["acc"]))

with ThreadPoolExecutor(8) as ex:
    verdicts = list(ex.map(lambda j: rejudge(j[2], j[3], j[4]), jobs))

results = {}
disagree = 0
for (i, step, q, gt, a, haiku), (opus, lines) in zip(jobs, verdicts):
    results.setdefault(i, {})[step] = (haiku, opus)
    if opus is not None and abs(opus - haiku) > 1e-6:
        disagree += 1
        reason = " ".join(lines)[:150]
        print(f"[idx {i} @{step}] haiku={haiku} opus={opus}  gt={gt[:50]!r} ans={a[:60]!r}")
        print(f"    opus: {reason}")

n = len(jobs)
print(f"\nagreement: {n - disagree}/{n} ({(n - disagree) / n:.0%})")

dh = sum(r["267"][0] - r["260"][0] for r in results.values())
do = sum(
    r["267"][1] - r["260"][1]
    for r in results.values()
    if r["267"][1] is not None and r["260"][1] is not None
)
print(f"acc delta over these {len(flips)} samples  haiku: {dh:+.1f} pts  opus: {do:+.1f} pts")
print(f"as val-acc delta (/114)               haiku: {dh / 114:+.4f}    opus: {do / 114:+.4f}")

harsh = sum(1 for r in results.values() if r["267"][1] is not None and r["267"][1] > r["267"][0])
lenient = sum(1 for r in results.values() if r["267"][1] is not None and r["267"][1] < r["267"][0])
print(f"on the 267 answers: opus grades higher than haiku on {harsh}, lower on {lenient}")
