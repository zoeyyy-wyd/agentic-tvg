#!/usr/bin/env bash
# Regenerate results/<run>/curves.png + metrics.csv for a GRPO run.
#
# This exists because nothing in training writes metrics.csv: run_grpo.sh passes
# trainer.logger=["console","tensorboard"], so verl only ever emits the console
# stream and the TB events. metrics.csv is a plot_grpo.py export, and it is only
# as fresh as the last time someone ran it -- on 2026-08-28 that left a csv
# stuck at step 67 while the run was at 212.
#
# Reads logs/tb/<exp>/ rather than a console log on purpose: plot_grpo.py merges
# every event file in the dir, last-write-wins per step, so the numbers survive a
# crash, a resume and an rm'd console log. Safe to run while training is live --
# read-only on logs/, and the plotter is a few seconds of CPU.
#
#   bash replot.sh                     # grpo_vanilla -> results/grpo-vanilla/
#   bash replot.sh grpo_smoke          # any logs/tb/<exp>
#   bash replot.sh grpo_vanilla /tmp/x # write somewhere else

set -euo pipefail
cd "$(dirname "$0")"

EXP="${1:-grpo_vanilla}"
TB="logs/tb/${EXP}"
# results/ spells with hyphens what verl's experiment_name spells with underscores.
OUT="${2:-results/${EXP//_/-}}"

die() { echo "错误: $*" >&2; exit 1; }

[ -d "${TB}" ] || die "没有 ${TB}
可选的实验: $(ls logs/tb 2>/dev/null | tr '\n' ' ')"
# plot_sft.py parses console logs only; the TB-directory reader is plot_grpo.py's.
case "${EXP}" in sft*) die "${EXP} 是 SFT 跑的，用: python plot_sft.py logs/<那个>.log --csv ${OUT}/metrics.csv" ;; esac
compgen -G "${TB}/events.out.tfevents.*" >/dev/null || die "${TB} 里没有 events 文件"

mkdir -p "${OUT}"
python plot_grpo.py "${TB}" -o "${OUT}/curves.png" --csv "${OUT}/metrics.csv"

# Same idiom as hf_push.sh: say what actually landed, not just that it landed.
python - "${OUT}/metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit("csv 是空的")
acc = next((c for c in rows[0] if c.startswith("val-core")), None)
print(f"steps 0-{rows[-1]['step']}, {len(rows)} 行")
pts = [(r["step"], float(r[acc])) for r in rows if acc and r[acc]]
if pts:
    print("val acc: " + "  ".join(f"{s}:{v:.3f}" for s, v in pts[-6:]))
PY
