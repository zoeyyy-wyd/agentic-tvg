#!/usr/bin/env python
"""Epoch-boundary curriculum: drop the prompts the policy has mastered.

Stage-1 GRPO (EPOCHS=1) visits each rl_train prompt exactly once with K=16
rollouts. A prompt whose visit came back with mean acc >= --threshold is very
likely to be pure saturation next epoch -- calibrated on round 1, where each
prompt got one visit per epoch: a >= 0.875 epoch-1 visit was mastered again in
epoch 2 85% of the time (52% with literally zero acc variance), and a
0.75-0.875 visit 57% of the time (GRPO_RESULTS §4). Cutting at 0.75 dropped
26.4% of prompts and would have cut epoch-2 mastered groups 23.1% -> 4.9%.

Keeps: every prompt below threshold, and every prompt the sampler never
reached (drop_last leaves a few unvisited per epoch). The hard end is kept on
purpose -- all-wrong groups still carry iou/partial-credit gradient.

Reads acc as recorded in the dump, i.e. whatever judge instrument the run
used; the threshold is applied on that same scale, so stage 2's cut must be
computed from stage 1's OWN rollouts, not round 1's (v1-judged) numbers.

Usage (between the two stages -- see GRPO2_PLAN §4):
  python data_prep/filter_mastered.py \
      --rollouts results/grpo-v2/rollouts --out data/processed/rl_train_ep2.parquet
It prints the exact EPOCHS/TOTAL_STEPS/TRAIN_FILE line to launch stage 2 with.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Same anchors as extract_rft.py's traceback: the question sits between fixed
# byte sequences of the rendered user prompt, and question text -> rl_train
# row is a verified unique key.
QUESTION_RE = re.compile(r'Question: "(.*)"\nAnswer the question based on the video\.', re.DOTALL)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", type=Path, default=Path("results/grpo-v2/rollouts"))
    ap.add_argument("--rl-train", type=Path, default=Path("data/processed/rl_train.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/rl_train_ep2.parquet"))
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="drop prompts whose (last) visit mean acc >= this")
    ap.add_argument("--max-step", type=int, default=0,
                    help="only use visits from steps <= this (0 = all; set it when testing "
                         "against a multi-epoch dump like round 1's)")
    ap.add_argument("--batch-size", type=int, default=8, help="prompts/step, for the stage-2 step math")
    args = ap.parse_args()

    files = sorted(args.rollouts.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"no <step>.jsonl under {args.rollouts}")
    visits: dict[str, list[tuple[int, float, int]]] = collections.defaultdict(list)
    max_step = 0
    for f in files:
        step = int(f.stem)
        if args.max_step and step > args.max_step:
            continue
        max_step = max(max_step, step)
        by_q: dict[str, list[float]] = collections.defaultdict(list)
        for line in open(f):
            r = json.loads(line)
            q = QUESTION_RE.search(r["input"])
            if q:
                by_q[q.group(1)].append(float(r["acc"]))
        for q, accs in by_q.items():
            visits[q].append((step, float(np.mean(accs)), len(accs)))

    # Last visit = the most recent policy's verdict on the prompt.
    last = {q: max(v)[1] for q, v in visits.items()}

    df = pd.read_parquet(args.rl_train)
    qs = [dict(r["extra_info"])["question"] for _, r in df.iterrows()]
    unmatched_rollout_qs = set(last) - set(qs)
    if unmatched_rollout_qs:
        raise SystemExit(f"{len(unmatched_rollout_qs)} rollout questions not in {args.rl_train} -- "
                         "wrong --rl-train for this run?")

    acc = np.array([last.get(q, -1.0) for q in qs])   # -1 = never visited -> kept
    keep = acc < args.threshold
    kept = df[keep]
    kept.to_parquet(args.out)

    sel = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "stage1_max_step": max_step,
        "rows": {"total": len(df), "kept": int(keep.sum()), "dropped": int((~keep).sum()),
                 "unvisited_kept": int((acc < 0).sum())},
        "dropped": sorted(
            ({"question": q, "visit_acc": round(last[q], 4)} for q, k in zip(qs, keep) if not k),
            key=lambda d: -d["visit_acc"]),
    }
    sel_path = args.out.with_suffix(".selection.json")
    sel_path.write_text(json.dumps(sel, indent=1, ensure_ascii=False))

    n2 = int(keep.sum()) // args.batch_size
    print(f"visited {len(last)}/{len(df)} prompts through step {max_step} | "
          f"dropped {int((~keep).sum())} (visit acc >= {args.threshold}) | "
          f"kept {int(keep.sum())} incl. {int((acc < 0).sum())} unvisited")
    print(f"-> {args.out} + {sel_path.name}")
    print(f"stage 2 ({n2} steps on the kept pool, resuming past step {max_step}):")
    print(f"  mv <ckpt>/global_step_{max_step}/data.pt{{,.bak}}   # see GRPO2_PLAN §4 -- MUST precede the resume")
    print(f"  EPOCHS=1 TOTAL_STEPS={max_step + n2} TRAIN_FILE={args.out} bash run_grpo.sh")


if __name__ == "__main__":
    main()
