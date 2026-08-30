#!/usr/bin/env bash
# The one SFT script — Qwen3-VL-4B + LoRA, 1x A100 80GB, verl's built-in SFT
# trainer (torchrun entry). Defaults run the QA main line (README).
#
#   SMOKE=1 bash run_sft.sh      # 2 real steps, no ckpt/eval          [~7 min]
#   bash run_sft.sh              # full QA SFT, ~2K rows x 2 epochs    [~6-8 h]
#
# Other data can be swapped in via TRAIN_FILES/VAL_FILES/EXP_NAME env vars.
#
# Key wiring, all verified against verl 0.9.0 + this repo's tests:
# - data.custom_cls -> agentic_tvg/sft_dataset.py::TVGMultiTurnSFTDataset
#   (fixes Qwen3-VL rope index + real video timestamps; see that file's docstring)
# - +data.apply_chat_template_kwargs.* -> processor decodes the video from its
#   path: 64 frames, fps disabled, whole-video pixel budget = per-frame x 64
#   ('+' prefix: the yaml default for apply_chat_template_kwargs is an empty dict)
# - max_length=20480: F=128 global view (~3.9K) + C=30 crops (up to ~4.6K each,
#   3 max) puts the longest trace near 19K; memory is governed by
#   max_token_len_per_gpu (24576 >= longest single trace) regardless
# - LoRA r=16 on the LLM only (ViT frozen via exclude_modules), lr 1e-4
#   (LongVT full-param used 5e-5; LoRA runs one notch hotter)
# - max_ckpt_to_keep=1: each checkpoint is ~19G (LoRA weights + optimizer state).
#   Unbounded, the ~5 save points of a full run need ~85G and fill the disk
#   mid-training -- that happened 2026-08-26 during the resume test (3 ckpts =
#   51G took / to 100%). Resume only ever needs the newest one.
#   CAVEAT, and it bites on every resume: verl tracks retention in an in-memory
#   list (checkpoint_manager.py, previous_saved_paths = []), so a resumed run
#   does not know about the checkpoint it resumed FROM and never deletes it.
#   Worse, max_ckpt_to_keep=1 saves before deleting (ensure_checkpoint_capacity
#   is a no-op below 2), so the next save needs 2x19G on top of that orphan.
#   After resuming, delete the checkpoint you resumed from by hand once the
#   next one is on disk -- verified 2026-08-26: resumed at 25, saved 50, both
#   survived, / hit 91% and step 75 would have died with ENOSPC.
#
# SFT-dose knob (plan §3 decision): SFT_DOSE=2000 for the light arm,
# unset/-1 for the full 6.1K arm.
set -xeuo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

# Guard the two silent killers before spending hours: wrong conda env, and the
# missing LD_PRELOAD that breaks torchcodec's first video decode (ENVIRONMENT.md §7).
set +x; source "${REPO}/env_setup/preflight.sh"; set -x

MODEL_PATH=${MODEL_PATH:-${REPO}/models/Qwen3-VL-4B-Instruct}
# Global-view frame knob. 128 is the production value (2026-08-26, was 64;
# coverage + sweep evidence in FRAMES_SWEEP.md) and MUST match the rendered
# data + system prompt (constants.GLOBAL_NUM_FRAMES) -- override only for
# memory-ceiling tests. Pixel budgets scale with it (whole-video budgets).
GLOBAL_FRAMES=${GLOBAL_FRAMES:-128}
LONGEST_EDGE=$((50176 * GLOBAL_FRAMES))
SHORTEST_EDGE=$((3136 * GLOBAL_FRAMES))
TRAIN_FILES=${TRAIN_FILES:-${REPO}/data/processed/sft_train.parquet}
VAL_FILES=${VAL_FILES:-${REPO}/data/processed/sft_val.parquet}
SFT_DOSE=${SFT_DOSE:--1}                 # -1 = all rows; e.g. 2000 = light arm
EPOCHS=${EPOCHS:-2}

SMOKE_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
    EXP_NAME=${EXP_NAME:-smoke}
    SMOKE_ARGS=(trainer.total_training_steps=2 trainer.save_freq=-1 trainer.test_freq=-1)
fi
EXP_NAME=${EXP_NAME:-sft_mix}

# results/<name>/ holds EVERYTHING this run generates (2026-08-30 reorg):
# ckpt/ + tb/ + console_<ts>.log attempts + the merged console.log, curves and
# config snapshot from the trap below. logs/ keeps only prepare_data's download
# logs. Hyphens by convention here (results/sft-mix), underscores in EXP_NAME;
# override RESULT_NAME to decouple the two.
RESULT_NAME=${RESULT_NAME:-${EXP_NAME//_/-}}
RESULT_DIR=${REPO}/results/${RESULT_NAME}

mkdir -p "${RESULT_DIR}"
# Structured metrics for every run: tensorboard event files in tb/, alongside
# the console log. curves.png is regenerated automatically on exit (see the
# trap below), crashed runs included.
export TENSORBOARD_DIR="${RESULT_DIR}/tb"

LOG_FILE="${RESULT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

# Plot on the way out, whatever the exit code -- the curve of a run that died
# is exactly when you want to see it. Every attempt of this experiment is
# concatenated oldest-first so a resumed run still yields one continuous curve;
# plot_sft keeps the LAST value of a repeated step, which is the one that
# survived the rollback. The [0-9] in the glob keeps this merged file (and any
# other hand-made log) from feeding itself back in.
plot_curves() {
    local merged="${RESULT_DIR}/console.log"
    # Array, not $(ls): an unmatched glob would leave `cat` with no arguments,
    # and cat with no arguments reads stdin -- the trap would hang the shell
    # instead of returning.
    local attempts=("${RESULT_DIR}"/console_[0-9]*.log)
    [ -e "${attempts[0]}" ] || return 0
    cat "${attempts[@]}" > "${merged}"
    python plot_sft.py "${merged}" -o "${RESULT_DIR}/curves.png" \
                                       --csv "${RESULT_DIR}/metrics.csv" || true
}
trap plot_curves EXIT

# Variable multimodal shapes (per-sample video grids + crop images, dynamic-bsz
# packing) fragment the CUDA caching allocator: QA smoke peaked at 75.6G
# *reserved* vs only 29.7G *allocated* on the 80G card. Expandable segments lets
# the allocator grow/shrink blocks instead of stranding them, reclaiming most of
# that gap. Standard torch>=2.1 setting; no effect on results.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}


torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples="${SFT_DOSE}" \
    data.custom_cls.path="${REPO}/agentic_tvg/sft_dataset.py" \
    data.custom_cls.name=Qwen3VLMultiTurnSFTDataset \
    +data.apply_chat_template_kwargs.num_frames="${GLOBAL_FRAMES}" \
    +data.apply_chat_template_kwargs.fps=null \
    +data.apply_chat_template_kwargs.videos_kwargs.size.longest_edge="${LONGEST_EDGE}" \
    +data.apply_chat_template_kwargs.videos_kwargs.size.shortest_edge="${SHORTEST_EDGE}" \
    data.max_length=20480 \
    data.truncation=error \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=24576 \
    model.path="${MODEL_PATH}" \
    model.lora_rank=16 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    model.exclude_modules='.*visual.*' \
    model.enable_gradient_checkpointing=True \
    optim.lr=1e-4 \
    optim.lr_warmup_steps_ratio=0.1 \
    trainer.project_name=agentic-tvg \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${RESULT_DIR}/ckpt" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.save_freq=25 \
    trainer.test_freq=25 \
    +trainer.max_ckpt_to_keep=1 \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} "$@" 2>&1 | tee "${LOG_FILE}"

# Config snapshot beside the result: hydra's fully-resolved config records the
# verl defaults this run actually used, which run_sft.sh alone does not show.
# `|| true` and `if`, not `&&`: on a fresh clone outputs/ does not exist yet,
# the glob stays literal, ls exits 2, and under `set -o pipefail -e` that would
# kill the script *after* a successful run -- or, as a trailing `&&` chain,
# hand back exit 1 from a run that worked.
_hydra=$(ls -1dt outputs/*/*/.hydra 2>/dev/null | head -1) || true
if [ -n "${_hydra:-}" ]; then
    cp "${_hydra}/config.yaml" "${RESULT_DIR}/hydra_config.yaml"
    cp "${_hydra}/overrides.yaml" "${RESULT_DIR}/hydra_overrides.yaml"
fi
