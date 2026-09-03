#!/usr/bin/env bash
# Stage-2 GRPO entry point: resume past the stage-1 checkpoint on the pruned
# pool (epoch-boundary curriculum, GRPO2_PLAN §3e). Thin wrapper over
# run_grpo.sh — same trainer, same defaults; only the horizon math and the
# dataloader-state surgery live here, both derived from disk state so the
# launch line cannot go stale:
#
#   steps_per_epoch = rows(TRAIN_FILE) // TRAIN_BS          (drop_last)
#   EPOCHS          = resumed_step // steps_per_epoch + 1   (verl skips
#       "already-run" epochs on resume: range(global_steps // len(loader),
#       total_epochs). EPOCHS=1 makes that range EMPTY on a smaller pool —
#       the run exits after val_before_train with zero training steps.
#       Measured the hard way 2026-09-03.)
#   TOTAL_STEPS     = resumed_step + steps_per_epoch        (one pass, capped)
#
# Also moves ckpt/global_step_<N>/data.pt aside: the saved dataloader state
# belongs to the OLD pool, and the restore's epoch-boundary check uses the
# NEW loader's length, so it would mis-restore instead of skipping.
#
# Usage:
#   bash run_grpo_stage2.sh                # after data_prep/filter_mastered.py
#   DRY=1 bash run_grpo_stage2.sh         # print the derived launch, run nothing

set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

TRAIN_FILE=${TRAIN_FILE:-${REPO}/data/processed/rl_train_ep2.parquet}
RESULT_DIR=${RESULT_DIR:-${REPO}/results/grpo-v2}
TRAIN_BS=${TRAIN_BS:-8}

[ -f "${TRAIN_FILE}" ] || { echo "missing ${TRAIN_FILE}" >&2
    echo "  build it: python data_prep/filter_mastered.py --rollouts ${RESULT_DIR}/rollouts" >&2; exit 1; }
[ -f "${RESULT_DIR}/ckpt/latest_checkpointed_iteration.txt" ] || {
    echo "no checkpoint registration under ${RESULT_DIR}/ckpt -- stage 1 not finished?" >&2; exit 1; }
STEP=$(tr -dc '0-9' < "${RESULT_DIR}/ckpt/latest_checkpointed_iteration.txt")
CKPT="${RESULT_DIR}/ckpt/global_step_${STEP}"
[ -d "${CKPT}" ] || { echo "checkpoint dir missing: ${CKPT}" >&2; exit 1; }

ROWS=$(python - "${TRAIN_FILE}" <<'PY'
import sys
import pyarrow.parquet as pq
print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)
PY
)
N2=$(( ROWS / TRAIN_BS ))
EPOCHS=$(( STEP / N2 + 1 ))

# The stage-2 horizon is anchored to the BOUNDARY step (where the pool was
# swapped), not to whatever checkpoint we happen to resume from -- a crash
# resume from, say, 140 must still end at boundary+N2, and its data.pt now
# belongs to the NEW pool and must NOT be moved aside. First launch records
# the boundary and does the surgery; later launches reuse it (2026-09-03: the
# naive STEP+N2 would have overshot and, worse, ended by epoch exhaustion
# before is_last_step -- no final save/val).
BOUNDARY_FILE="${RESULT_DIR}/stage2_boundary.txt"
if [ ! -f "${BOUNDARY_FILE}" ]; then
    if [ -f "${CKPT}/data.pt" ]; then
        mv "${CKPT}/data.pt" "${CKPT}/data.pt.bak"
        echo "[stage2] boundary launch: dataloader state moved aside (${CKPT}/data.pt -> .bak)"
    fi
    echo "${STEP}" > "${BOUNDARY_FILE}"
fi
BOUNDARY=$(tr -dc '0-9' < "${BOUNDARY_FILE}")
TOTAL=$(( BOUNDARY + N2 ))

echo "[stage2] resume past step ${STEP} (boundary ${BOUNDARY}) | pool ${ROWS} rows -> ${N2} steps/epoch | EPOCHS=${EPOCHS} TOTAL_STEPS=${TOTAL}"
[ "${DRY:-0}" = "1" ] && exit 0
# val_before_train off for resumes (user call 2026-09-03): the step-<N> val was
# already produced by the finishing stage, re-running it buys ~20 min of GPU
# for a number we have. Hydra takes the last occurrence of a duplicate key, so
# this overrides run_grpo.sh's True; trailing "$@" can override back.
EPOCHS="${EPOCHS}" TOTAL_STEPS="${TOTAL}" TRAIN_FILE="${TRAIN_FILE}"     exec bash run_grpo.sh trainer.val_before_train=False "$@"
