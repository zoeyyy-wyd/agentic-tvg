#!/usr/bin/env bash
# Charades-STA zero-shot grounding probe: compare recipe stages on an EXTERNAL
# short-video temporal-grounding benchmark (~30s videos, well inside the F=128
# budget), without changing the answer format the models were trained on --
# each row asks a natural "when does X happen" question, the policy crops
# where it believes the moment is, and the metric is evidence_iou =
# IoU(crop window, GT span), computed by the same reward code as training.
# Judge disabled (no QA ground truth): zero API cost, pure IoU + recall.
#
# First run (2026-09-05, 399 queries / 328 videos, seed 0):
#   sft-mix 0.379 mIoU / 37.3% R@0.5 -> grpo-v2 0.482 / 52.4% -> rft-v2b 0.475 / 46.6%
#
# Re-run safe: every stage is skipped if its output already exists. ~1h GPU
# per model when evals actually run. Check /tmp/gpu-owner.lock first.
#
# Usage:
#   bash run_charades_probe.sh                       # all three stages + table
#   MODELS="grpo-v2" bash run_charades_probe.sh      # subset
#   N=400 bash run_charades_probe.sh                 # sample size (seed 0, deterministic)

set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)
source "${REPO}/env_setup/preflight.sh"

MODELS=${MODELS:-"sft-mix grpo-v2 rft-v2b"}
N=${N:-400}
ANN_DIR=data/annotations/charades
VID_DIR=data/videos/charades
PROBE=data/processed/charades_probe.parquet
ANN_URL="https://raw.githubusercontent.com/microsoft/VideoX/master/2D-TAN/data/Charades-STA/charades_sta_test.txt"
ZIP_URL="https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"

step() { echo -e "\n\033[1;32m==> $*\033[0m"; }
mkdir -p "${ANN_DIR}" "${VID_DIR}" data/archives data/processed

step "1/5 annotations"
[ -f "${ANN_DIR}/charades_sta_test.txt" ] || curl -sL --retry 3 "${ANN_URL}" -o "${ANN_DIR}/charades_sta_test.txt"

step "2/5 deterministic sample (seed 0, N=${N})"
if [ ! -f "${ANN_DIR}/sample${N}.txt" ]; then
python - "$N" <<'PY'
import sys
import numpy as np
n = int(sys.argv[1])
rows = [l.strip() for l in open("data/annotations/charades/charades_sta_test.txt") if l.strip()]
idx = np.random.default_rng(0).choice(len(rows), size=n, replace=False)
open(f"data/annotations/charades/sample{n}.txt", "w").write("\n".join(rows[i] for i in sorted(idx)))
vids = sorted({rows[i].split("##")[0].split()[0] for i in idx})
open(f"data/annotations/charades/sample{n}_vids.txt", "w").write("\n".join(vids))
print(f"{n} queries over {len(vids)} videos")
PY
fi

step "3/5 videos (13G zip streamed once, only sampled mp4s kept)"
MISSING=$(comm -23 <(sort "${ANN_DIR}/sample${N}_vids.txt") <(ls "${VID_DIR}" 2>/dev/null | sed 's/\.mp4$//' | sort) | wc -l)
if [ "${MISSING}" -gt 0 ]; then
    echo "${MISSING} videos missing -- fetching zip (resumable; aria2c would parallelize if installed)"
    curl -sL -C - --retry 3 -o data/archives/Charades_v1_480.zip "${ZIP_URL}"
    while read -r v; do
        [ -f "${VID_DIR}/${v}.mp4" ] || unzip -j -o -q data/archives/Charades_v1_480.zip "Charades_v1_480/${v}.mp4" -d "${VID_DIR}" 2>/dev/null || true
    done < "${ANN_DIR}/sample${N}_vids.txt"
    rm -f data/archives/Charades_v1_480.zip
fi
echo "videos on disk: $(ls "${VID_DIR}" | wc -l)"

step "4/5 probe parquet"
[ -f "${PROBE}" ] || python data_prep/prepare_charades.py --ann "${ANN_DIR}/sample${N}.txt" --videos-dir "${VID_DIR}" --out "${PROBE}"

step "5/5 evals (skipped where the dump already exists)"
export JUDGE_DISABLE=1
for M in ${MODELS}; do
    BENCH_DIR="results/bench-charades-${M}"
    if ls "${BENCH_DIR}"/val_rollouts/*.jsonl >/dev/null 2>&1; then
        echo "  [skip] ${M} -- dump exists"
        continue
    fi
    [ -d "results/${M}/merged" ] || { echo "missing results/${M}/merged" >&2; exit 1; }
    MODEL_PATH="results/${M}/merged" EXP_NAME="bench_charades_${M//-/_}" \
        VAL_FILE="${PROBE}" bash run_grpo.sh trainer.val_only=True
done

step "summary"
python - ${MODELS} <<'PY'
import glob
import json
import sys

import numpy as np

print(f"{'model':>10} | {'n':>3} | {'mean IoU':>8} | {'R@0.3':>6} | {'R@0.5':>6} | {'R@0.7':>6} | {'tools':>5}")
for m in sys.argv[1:]:
    fps = sorted(glob.glob(f"results/bench-charades-{m}/val_rollouts/*.jsonl"))
    if not fps:
        print(f"{m:>10} | (no dump)")
        continue
    rows = [json.loads(l) for l in open(fps[0])]
    iou = np.array([r["evidence_iou"] for r in rows])
    tc = np.array([r["num_tool_calls"] for r in rows])
    print(f"{m:>10} | {len(rows):>3} | {iou.mean():>8.4f} | {np.mean(iou>=0.3)*100:>5.1f}% "
          f"| {np.mean(iou>=0.5)*100:>5.1f}% | {np.mean(iou>=0.7)*100:>5.1f}% | {tc.mean():>5.2f}")
PY
