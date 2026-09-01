#!/usr/bin/env bash
# Stage-3 RFT entry point: SFT on the merged GRPO model over the policy's own
# filtered rollouts (data recipe: DATA.md §8; build: data_prep/extract_rft.py).
#
# Deliberately a thin wrapper: RFT *is* the run_sft.sh machinery — same
# trainer, LoRA config, loss mask — and duplicating the torchrun block here
# would let the two copies drift apart silently. What makes RFT a different
# stage is only WHAT to train on and FROM WHERE, and those three deltas are
# pinned below. Anything run_sft.sh accepts still works: env overrides win
# over the defaults here, and trailing args pass through to hydra.
#
# Usage:
#   bash run_rft.sh                          # RFT -> results/rft-v2/
#   SMOKE=1 bash run_rft.sh                  # 2-step smoke -> results/smoke/ (rm after)
#   MODEL_PATH=... bash run_rft.sh trainer.total_epochs=1

set -euo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

# The three deltas vs plain SFT. Exported because run_sft.sh runs as a child.
# All three point at round 2 since 2026-09-01; round 1's stage-3 run is
# reproduced with TRAIN_FILES/VAL_FILES=.../rft_{train,val}.parquet,
# MODEL_PATH=results/grpo-vanilla/merged, EXP_NAME=rft.
export TRAIN_FILES=${TRAIN_FILES:-${REPO}/data/processed/rft_v2_train.parquet}
export VAL_FILES=${VAL_FILES:-${REPO}/data/processed/rft_v2_val.parquet}
export MODEL_PATH=${MODEL_PATH:-${REPO}/results/grpo-v2/merged}
# Under SMOKE=1, leave EXP_NAME for run_sft.sh to default to "smoke" —
# pinning "rft_v2" here would write smoke logs/tb into results/rft-v2/.
if [ "${SMOKE:-0}" != "1" ]; then
    export EXP_NAME=${EXP_NAME:-rft_v2}
fi

# Fail in seconds, not 20 minutes in: both inputs are products of earlier
# stages, and each error names the command that builds the missing one.
[ -f "${TRAIN_FILES}" ] || { echo "TRAIN_FILES missing: ${TRAIN_FILES}" >&2
    echo "  build it: python data_prep/extract_rft.py   (reads results/grpo-v2/rollouts/)" >&2
    exit 1; }
[ -d "${MODEL_PATH}" ] || { echo "MODEL_PATH missing: ${MODEL_PATH}" >&2
    echo "  build it: python merge_adapter.py --ckpt results/grpo-v2/ckpt \\" >&2
    echo "                --out results/grpo-v2/merged --base results/sft-mix/merged" >&2
    exit 1; }

exec bash run_sft.sh "$@"
