#!/usr/bin/env python
"""Per-question / curriculum figure: accuracy histogram, variance-vs-difficulty, saturation.

Three panels from the group-level facts GRPO actually trains on:

1. histogram of per-question accuracy (mean acc over the K rollouts of each
   question's visit), 0.1-wide bins with counts, split at the curriculum cut;
2. within-group reward variance vs question accuracy -- the ∩ shape: variance
   (= the advantage's raw material) peaks at mid difficulty and dies at both
   ends, which is the visual argument for cutting the mastered spike;
3. pool saturation over training (share of visited groups with mean acc >=
   0.9), v2 vs the v1 baseline.

Inputs: the CSV from `analyze_groups.py --per-question` (panels 1-2) and the
two rollout dirs (panel 3).

Usage:
  python plot_per_question.py                              # stage-1 defaults
  python plot_per_question.py --cut 0.9 -o results/grpo-v2/per_question_analysis.png
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEPT, DROP, INK, MUT = "#3465c0", "#e07b39", "#222222", "#777777"


def mastered_by_band(rollout_glob: str, bands) -> tuple[list, list]:
    xs, ys = [], []
    for lo, hi in bands:
        accs = []
        for f in glob.glob(rollout_glob):
            step = int(Path(f).stem)
            if not lo <= step < hi:
                continue
            by_prompt = collections.defaultdict(list)
            for line in open(f):
                r = json.loads(line)
                by_prompt[r["input"]].append(r["acc"])
            accs += [np.mean(v) for v in by_prompt.values()]
        if accs:
            xs.append((lo + min(hi, 270)) / 2)
            ys.append(np.mean(np.array(accs) >= 0.9) * 100)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-question", type=Path, default=Path("results/grpo-v2/per_question_stage1.csv"))
    ap.add_argument("--v2-rollouts", default="results/grpo-v2/rollouts/*.jsonl")
    ap.add_argument("--v1-rollouts", default="results/grpo-vanilla/rollouts_grpo267/*.jsonl")
    ap.add_argument("--cut", type=float, default=0.9, help="curriculum threshold drawn on panels 1-2")
    ap.add_argument("--unvisited", type=int, default=4, help="prompts the sampler never reached (kept)")
    ap.add_argument("-o", "--out", type=Path, default=Path("results/grpo-v2/per_question_analysis.png"))
    args = ap.parse_args()

    pq = pd.read_csv(args.per_question)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    fig.subplots_adjust(wspace=0.28, left=0.05, right=0.985, top=0.86, bottom=0.14)

    # --- 1. accuracy distribution -----------------------------------------
    ax = axes[0]
    bins = np.arange(0, 1.01, 0.1)
    kept = pq[pq.acc_mean < args.cut]["acc_mean"]
    dropped = pq[pq.acc_mean >= args.cut]["acc_mean"]
    ax.hist(kept, bins=bins, color=KEPT, edgecolor="white", linewidth=1.2,
            label=f"kept for next stage (n={len(kept)}+{args.unvisited} unvisited)")
    ax.hist(dropped, bins=bins, color=DROP, edgecolor="white", linewidth=1.2,
            label=f"dropped, mastered (n={len(dropped)})")
    counts, _ = np.histogram(pq.acc_mean, bins=bins)
    for i, cnt in enumerate(counts):
        ax.text(bins[i] + 0.05, cnt + 3, str(cnt), ha="center", fontsize=8, color=MUT)
    ax.axvline(args.cut, color=INK, ls="--", lw=1.2)
    ax.text(args.cut - 0.012, ax.get_ylim()[1] * 0.88, f"cut: visit acc ≥ {args.cut} ",
            ha="right", va="top", fontsize=9, color=INK)
    ax.set_xticks(bins)
    ax.set_xlabel("question accuracy over its 16 rollouts (acc_mean)")
    ax.set_ylabel("questions")
    ax.set_title(f"v2: per-question accuracy ({len(pq):,} questions)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- 2. group variance vs difficulty ----------------------------------
    ax = axes[1]
    m = pq.acc_mean >= args.cut
    ax.axvspan(args.cut, 1.02, color=DROP, alpha=0.08)
    ax.scatter(pq.acc_mean[~m], pq.score_var[~m], s=14, color=KEPT, alpha=0.45, linewidths=0, label="kept")
    ax.scatter(pq.acc_mean[m], pq.score_var[m], s=14, color=DROP, alpha=0.45, linewidths=0, label="dropped")
    med = pq.groupby(pd.cut(pq.acc_mean, np.arange(0, 1.01, 0.125), include_lowest=True),
                     observed=True)["score_var"].median()
    ax.plot([iv.mid for iv in med.index], med.values, color=INK, lw=2, label="median")
    ax.annotate("no gradient here\n(all 16 agree)", xy=(0.99, 0.01), xytext=(0.64, 0.44),
                fontsize=9, color=MUT, arrowprops=dict(arrowstyle="->", color=MUT, lw=1))
    ax.annotate("GRPO's signal lives here", xy=(0.42, float(pq.score_var.max()) * 0.9),
                ha="center", fontsize=9, color=MUT)
    ax.set_xlabel("question accuracy (acc_mean)")
    ax.set_ylabel("within-group reward variance (score_var)")
    ax.set_title("Group variance vs difficulty — the ∩ shape", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # --- 3. saturation, v2 vs v1 ------------------------------------------
    ax = axes[2]
    x1, y1 = mastered_by_band(args.v1_rollouts,
                              [(1, 40), (40, 80), (80, 120), (120, 160), (160, 200), (200, 240), (240, 268)])
    x2, y2 = mastered_by_band(args.v2_rollouts, [(1, 40), (40, 80), (80, 120), (120, 134)])
    ax.plot(x1, y1, color=DROP, lw=2, marker="o", ms=5)
    ax.plot(x2, y2, color=KEPT, lw=2, marker="o", ms=5)
    ax.text(x1[-1] - 6, y1[-1] - 2.6, "v1 (grpo-vanilla, 267 steps)", color=DROP, fontsize=9, ha="right")
    ax.text(x2[-1] + 5, y2[-1], "v2 stage 1", color=KEPT, fontsize=9, va="center")
    ax.axhline(30, color=MUT, lw=0.8, ls=":")
    ax.text(3, 30.6, "~30% mastered", fontsize=8, color=MUT)
    ax.set_xlabel("training step")
    ax.set_ylabel("mastered questions in batch (%)  [mean acc ≥ 0.9]")
    ax.set_title("Pool saturation: v2 reaches it at 2× v1's speed", fontsize=11)
    ax.set_ylim(0, 36)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
