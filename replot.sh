#!/usr/bin/env bash
# Regenerate results/<run>/curves.png + metrics.csv from that run's tb/.
#
# Since the 2026-08-30 reorg both run scripts do this themselves in an EXIT
# trap, so this exists for looking at a run that is still going (the trap only
# fires when it ends). Reads results/<run>/tb rather than a console log:
# plot_grpo.py merges every event file in the dir, last-write-wins per step,
# so the numbers survive a crash, a resume and an rm'd console log. Safe while
# training is live -- read-only on tb/, a few seconds of CPU.
#
#   bash replot.sh                     # results/grpo-vanilla/
#   bash replot.sh grpo-smoke          # any results/<run> with a tb/
#   bash replot.sh grpo_vanilla        # underscores accepted, hyphenated for you

set -euo pipefail
cd "$(dirname "$0")"

RUN="${1:-grpo-vanilla}"
RUN="${RUN//_/-}"                      # EXP_NAME spelling -> results/ spelling
OUT="results/${RUN}"
TB="${OUT}/tb"

die() { echo "error: $*" >&2; exit 1; }

[ -d "${TB}" ] || die "no ${TB}
runs with a tb/: $(ls -d results/*/tb 2>/dev/null | sed 's|results/||;s|/tb||' | tr '\n' ' ')"
# plot_sft.py parses console logs only; the TB-directory reader is plot_grpo.py's.
case "${RUN}" in sft*) die "${RUN} is an SFT run; use: python plot_sft.py ${OUT}/console.log --csv ${OUT}/metrics.csv" ;; esac
compgen -G "${TB}/events.out.tfevents.*" >/dev/null || die "no events files in ${TB}"

python plot_grpo.py "${TB}" -o "${OUT}/curves.png" --csv "${OUT}/metrics.csv"

# Same idiom as hf_push.sh: say what actually landed, not just that it landed.
python - "${OUT}/metrics.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    sys.exit("csv is empty")
acc = next((c for c in rows[0] if c.startswith("val-core")), None)
print(f"steps 0-{rows[-1]['step']}, {len(rows)} rows")
pts = [(r["step"], float(r[acc])) for r in rows if acc and r[acc]]
if pts:
    print("val acc: " + "  ".join(f"{s}:{v:.3f}" for s, v in pts[-6:]))
PY
