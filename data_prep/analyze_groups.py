#!/usr/bin/env python
"""Group-level analysis of TRAINING rollouts: where GRPO's gradient comes from.

Where analyze_rollouts.py works row-by-row on val/benchmark dumps, this works
on the *group* -- the K rollouts of one prompt, which is the only unit GRPO's
advantage is defined over. Two questions:

1. `--signal`: does the within-group signal survive the run? Per step band:
   the share of groups the policy has mastered, the share with zero acc
   variance, and the mean within-group reward spread. A pool that saturates
   loses gradient even while lr, batch and K are untouched (DATA.md §3
   predicted exactly this failure mode for the 1,068-prompt pool).

2. `--metrics metrics.csv`: learning speed against lr. Fits the slope of
   train score and policy entropy over three phases and prints the mean lr of
   each. The pairing matters: entropy slope alone says how fast the policy is
   sharpening, not whether it is sharpening usefully, so it is only read
   next to the score slope.

3. `--reward-ab`: would a different reward have ranked the SAME rollouts
   differently? Re-scores every recorded trajectory under a candidate reward
   and reports the within-group Spearman of the two advantage vectors, how
   often the winning trajectory changes, and how many groups lose their
   gradient. This is the cheap pre-flight for a reward change: a candidate
   that leaves the ranking at rho ~ 1.0 cannot change what the policy learns,
   whatever its motivation looks like on paper.

   The judge variant is a SIMULATION, not a prediction: PARTIAL verdicts are
   reassigned FULL/INCORRECT at --partial-to-full, the split measured by the
   2026-09-01 opus audit (11 -> FULL, 5 -> 0). Real v2 verdicts are
   systematic where this is random, so read the number as an effect size
   against the other candidates, not as an expected outcome.

Usage:
  python data_prep/analyze_groups.py results/grpo-vanilla/rollouts_grpo267 --signal \
      --metrics results/grpo-vanilla/metrics.csv
  python data_prep/analyze_groups.py results/grpo-vanilla/rollouts_grpo267 \
      --reward-ab --time-weights 1.0 2.0 5.0 --partial-to-full 0.69
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

BANDS = [(1, 40), (40, 80), (80, 120), (120, 160), (160, 200), (200, 240), (240, 10**9)]

QUESTION_RE = re.compile(r'Question: "(.*)"\nAnswer the question based on the video\.', re.DOTALL)


def load_groups(rollouts: Path) -> list[tuple[int, list[dict]]]:
    """[(step, [row] * K)] -- grouped by prompt text, which is GRPO's group key."""
    files = sorted(glob.glob(str(rollouts / "*.jsonl")), key=lambda p: int(Path(p).stem))
    if not files:
        raise SystemExit(f"no <step>.jsonl under {rollouts}")
    out = []
    for f in files:
        by_prompt: dict[str, list[dict]] = collections.defaultdict(list)
        for line in open(f):
            r = json.loads(line)
            by_prompt[r["input"]].append(r)
        for g in by_prompt.values():
            out.append((g[0]["step"], g))
    return out


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared. numpy-only so the script runs in the verl
    env, which has no scipy."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(len(a), dtype=float)
    srt = a[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def advantage(r: np.ndarray) -> np.ndarray:
    """GRPO's group-normalized advantage. Zero std -> zero gradient."""
    s = r.std()
    return np.zeros_like(r) if s < 1e-9 else (r - r.mean()) / s


def report_signal(groups, mastered_at: float) -> None:
    rows = collections.defaultdict(list)
    for step, g in groups:
        band = next((b for b in BANDS if b[0] <= step < b[1]), None)
        if band is None:
            continue
        acc = np.array([r["acc"] for r in g])
        rows[band].append((acc.mean(), acc.std(), np.array([r["score"] for r in g]).std()))
    print(f"{'steps':>10} | {'groups':>6} | {'all-wrong':>9} | {f'mastered(>={mastered_at})':>16} "
          f"| {'zero acc-var':>12} | {'score std':>9}")
    for band in BANDS:
        a = np.array(rows[band]) if rows[band] else None
        if a is None or not len(a):
            continue
        hi = "+" if band[1] > 10**8 else str(band[1])
        print(f"{band[0]:>4}-{hi:<5} | {len(a):>6} | {(a[:, 0] == 0).mean() * 100:>8.1f}% "
              f"| {(a[:, 0] >= mastered_at).mean() * 100:>15.1f}% "
              f"| {(a[:, 1] < 1e-9).mean() * 100:>11.1f}% | {a[:, 2].mean():>9.4f}")


def report_reward_ab(groups, time_weights, p_full, seeds) -> None:
    """Round-1 reward vs candidates, on round 1's own trajectories."""
    def compare(mutate, label):
        rho, top, n, died = [], 0, 0, 0
        for _, g in groups:
            acc = np.array([r["acc"] for r in g])
            iou = np.array([r["evidence_iou"] for r in g])
            fmt = np.array([r["format_score"] for r in g])
            r1 = fmt + acc + 0.5 * iou
            r2 = mutate(acc, iou, fmt)
            if r1.std() < 1e-9 or r2.std() < 1e-9:
                died += r2.std() < 1e-9 <= r1.std()
                continue
            n += 1
            c = spearman(advantage(r1), advantage(r2))
            if not np.isnan(c):
                rho.append(c)
            top += int(np.argmax(r1)) != int(np.argmax(r2))
        print(f"{label:>34} | {np.mean(rho):>8.3f} | {top / n * 100:>10.1f}% | {died / len(groups) * 100:>9.1f}%")

    print(f"{'candidate reward':>34} | {'Spearman':>8} | {'top flips':>11} | {'died':>9}")
    for w in time_weights:
        compare(lambda acc, iou, fmt, w=w: fmt + acc + w * iou, f"TIME_WEIGHT {0.5} -> {w}")
    for sd in seeds:
        rng = np.random.default_rng(sd)

        def judge(acc, iou, fmt, rng=rng):
            a = acc.copy()
            m = acc == 0.5
            a[m] = (rng.random(m.sum()) < p_full).astype(float)
            return fmt + a + 0.5 * iou
        compare(judge, f"judge v2 sim (seed {sd})")


PHASES = [(1, 90), (90, 180), (180, 10**9)]


def report_metrics(path: Path) -> None:
    """Learning speed vs lr, from a run's metrics.csv."""
    import csv

    rows = list(csv.DictReader(open(path)))

    def col(r, k):
        v = r.get(k, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    pts = [(int(float(r["step"])), col(r, "actor/lr"), col(r, "critic/score/mean"),
            col(r, "actor/entropy"), col(r, "actor/grad_norm")) for r in rows if r.get("step")]
    pts = [p for p in pts if all(x is not None for x in p[1:])]
    print(f"{'phase':>12} | {'mean lr':>9} | {'score slope':>12} | {'entropy slope':>14} | {'grad_norm':>9}")
    for lo, hi in PHASES:
        m = [p for p in pts if lo <= p[0] < hi]
        if len(m) < 3:
            continue
        step = np.array([p[0] for p in m], float)
        lr, sc, en, gn = (np.array([p[i] for p in m], float) for i in (1, 2, 3, 4))
        hi_s = "+" if hi > 10**8 else str(hi)
        print(f"{lo:>5}-{hi_s:<6} | {lr.mean():>9.2e} | {np.polyfit(step, sc, 1)[0] * 100:>+12.4f} "
              f"| {np.polyfit(step, en, 1)[0] * 100:>+14.4f} | {gn.mean():>9.4f}")
    print("  (slopes are per 100 steps)")


def report_per_question(groups, out: Path) -> None:
    """One row per (question, visit): the group-level facts GRPO actually
    trains on -- mean acc over the K rollouts (the question's accuracy), the
    within-group variance of the full reward (the advantage's raw material),
    and the component stats. Also the curriculum's decision table: sort by
    acc_mean and the filter_mastered cut is the head of the file."""
    rows = []
    for step, g in groups:
        acc = np.array([r["acc"] for r in g])
        sc = np.array([r["score"] for r in g])
        iou = np.array([r["evidence_iou"] for r in g])
        q = QUESTION_RE.search(g[0]["input"])
        rows.append({
            "question": q.group(1) if q else "?",
            "step": step, "k": len(g),
            "acc_mean": round(acc.mean(), 4), "acc_std": round(acc.std(), 4),
            "n_full": int((acc == 1.0).sum()), "n_partial": int((acc == 0.5).sum()),
            "n_wrong": int((acc == 0.0).sum()),
            "score_mean": round(sc.mean(), 4), "score_var": round(sc.var(), 5),
            "score_std": round(sc.std(), 4),
            "iou_mean": round(iou.mean(), 4), "iou_max": round(iou.max(), 4),
            "zero_acc_var": int(acc.std() < 1e-9), "zero_score_var": int(sc.std() < 1e-9),
        })
    rows.sort(key=lambda r: (-r["acc_mean"], -r["score_var"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    accm = np.array([r["acc_mean"] for r in rows]); sv = np.array([r["score_var"] for r in rows])
    print(f"per-question: {len(rows)} visits -> {out}")
    print(f"  acc_mean quartiles {np.percentile(accm,[25,50,75]).round(3).tolist()} | mastered(>=0.9) "
          f"{(accm>=0.9).mean()*100:.1f}% | >=0.75 {(accm>=0.75).mean()*100:.1f}% | all-wrong {(accm==0).mean()*100:.1f}%")
    print(f"  score_var quartiles {np.percentile(sv,[25,50,75]).round(4).tolist()} | zero-acc-var "
          f"{np.mean([r['zero_acc_var'] for r in rows])*100:.1f}% | zero-score-var {np.mean([r['zero_score_var'] for r in rows])*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts", type=Path, help="a run's rollouts/ dir of <step>.jsonl")
    ap.add_argument("--signal", action="store_true")
    ap.add_argument("--reward-ab", action="store_true")
    ap.add_argument("--metrics", type=Path, help="a run's metrics.csv: learning speed vs lr")
    ap.add_argument("--per-question", type=Path, help="write one row per (question, visit) to this CSV")
    ap.add_argument("--mastered-at", type=float, default=0.9, help="group mean acc counted as mastered")
    ap.add_argument("--time-weights", type=float, nargs="*", default=[1.0, 2.0, 5.0])
    ap.add_argument("--partial-to-full", type=float, default=0.69, help="PARTIAL->FULL share in the judge sim")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    groups = load_groups(args.rollouts)
    ks = collections.Counter(len(g) for _, g in groups)
    print(f"{len(groups)} groups, {sum(len(g) for _, g in groups)} trajectories, group sizes {dict(ks)}\n")
    if args.signal or not args.reward_ab:
        report_signal(groups, args.mastered_at)
        print()
    if args.metrics:
        report_metrics(args.metrics)
        print()
    if args.per_question:
        report_per_question(groups, args.per_question)
        print()
    if args.reward_ab:
        report_reward_ab(groups, args.time_weights, args.partial_to_full, args.seeds)


if __name__ == "__main__":
    main()
