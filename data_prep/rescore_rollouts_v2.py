#!/usr/bin/env python
"""Re-score dumped rollouts under the round-2 instrument + reward, and save.

Purpose: the endgame comparison (GRPO2_PLAN §5) is a 3-checkpoint mean vs
3-checkpoint mean on ONE scale. Round-1's val dumps carry v1-judged acc from
rollout time; this re-grades the SAME answers with the live v2 judge (verdicts
cache into judge_cache_v2.jsonl like any other v2 verdict -- re-runs are free)
and recomputes the round-2 reward from the dump's recorded deterministic
components:

    reward_v2 = format_score(recorded) + acc_v2 + TIME_WEIGHT_V2 * evidence_iou(recorded)

Only the instrument changes; the generated text is untouched. format_score and
evidence_iou are pure functions of that text (computed by the same code at
rollout time), so they are reused, not recomputed.

Usage (no GPU; needs ANTHROPIC_API_KEY via .env):
  python data_prep/rescore_rollouts_v2.py \
      results/grpo-vanilla/val_rollouts_grpo267/240.jsonl \
      results/grpo-vanilla/val_rollouts_grpo267/260.jsonl \
      results/grpo-vanilla/val_rollouts_grpo267/267.jsonl \
      --out results/grpo-vanilla/v2_rescore.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentic_tvg.answer_match import parse_answer_qa          # noqa: E402
from agentic_tvg.judge_v2 import JUDGE_MODEL, judge_answer    # noqa: E402
from agentic_tvg.reward import TIME_WEIGHT_V2                 # noqa: E402

QUESTION_RE = re.compile(r'Question: "(.*)"\nAnswer the question based on the video\.', re.DOTALL)


def rescore_row(r: dict) -> dict:
    q = QUESTION_RE.search(r["input"])
    parsed = parse_answer_qa(r["output"] or "")
    acc_v2 = 0.0
    if q and parsed.answer:
        v = judge_answer(q.group(1), str(r["gts"]), parsed.answer)
        acc_v2 = float(v) if v is not None else 0.0
    fmt, iou = float(r["format_score"]), float(r["evidence_iou"])
    return {
        "question": q.group(1) if q else None,
        "answer": parsed.answer,
        "acc_v1_recorded": float(r["acc"]),
        "acc_v2": acc_v2,
        "format_score": fmt,
        "evidence_iou": iou,
        "reward_v2": fmt + acc_v2 + TIME_WEIGHT_V2 * iou,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    result = {"judge_model": JUDGE_MODEL, "rubric": "v2", "time_weight_v2": TIME_WEIGHT_V2, "files": {}}
    for f in args.jsonl:
        rows = [json.loads(l) for l in open(f)]
        with ThreadPoolExecutor(args.threads) as ex:
            scored = list(ex.map(rescore_row, rows))
        n = len(scored)
        means = {k: sum(s[k] for s in scored) / n
                 for k in ("acc_v1_recorded", "acc_v2", "format_score", "evidence_iou", "reward_v2")}
        result["files"][str(f)] = {"n": n, "means": means, "rows": scored}
        print(f"{f}: n={n}  acc v1->v2 {means['acc_v1_recorded']:.4f} -> {means['acc_v2']:.4f}  "
              f"reward_v2 {means['reward_v2']:.4f}", flush=True)

    fm = [v["means"] for v in result["files"].values()]
    result["mean_of_checkpoints"] = {k: sum(m[k] for m in fm) / len(fm) for k in fm[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    mc = result["mean_of_checkpoints"]
    print(f"\n{len(fm)}-checkpoint mean: acc_v2 {mc['acc_v2']:.4f}  reward_v2 {mc['reward_v2']:.4f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
