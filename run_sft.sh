#!/usr/bin/env bash
# The one SFT script — Qwen3-VL-4B + LoRA, 1x A100 80GB, verl's built-in SFT
# trainer (torchrun entry). Defaults run the QA main line (PLAN.md).
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
#   path: 32 frames, fps disabled, whole-video pixel budget = per-frame x 32
#   ('+' prefix: the yaml default for apply_chat_template_kwargs is an empty dict)
# - max_length=12288: QA traces run longer than the old TVG set (its val max was
#   4118 tokens); memory is governed by max_token_len_per_gpu regardless
# - LoRA r=16 on the LLM only (ViT frozen via exclude_modules), lr 1e-4
#   (LongVT full-param used 5e-5; LoRA runs one notch hotter)
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

mkdir -p ckpts logs
# Structured metrics for every run: tensorboard event files under logs/tb/,
# alongside the console log. Plot PNGs afterwards with: python plot_metrics.py
export TENSORBOARD_DIR="${REPO}/logs/tb/${EXP_NAME}"

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
    +data.apply_chat_template_kwargs.num_frames=32 \
    +data.apply_chat_template_kwargs.fps=null \
    +data.apply_chat_template_kwargs.videos_kwargs.size.longest_edge=1605632 \
    +data.apply_chat_template_kwargs.videos_kwargs.size.shortest_edge=100352 \
    data.max_length=12288 \
    data.truncation=error \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=1 \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=16384 \
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
    trainer.default_local_dir="${REPO}/ckpts/${EXP_NAME}" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.logger='["console","tensorboard"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} "$@" 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
