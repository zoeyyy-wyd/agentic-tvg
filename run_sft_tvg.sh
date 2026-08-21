#!/usr/bin/env bash
# SFT Stage-1 (plan §5) — Qwen3-VL-4B + LoRA on re-rendered LongVT tvg traces,
# 1x A100 80GB, verl's built-in SFT trainer (torchrun entry).
#
# Key wiring, all verified against verl 0.9.0 + this repo's tests:
# - data.custom_cls -> agentic_tvg/sft_dataset.py::TVGMultiTurnSFTDataset
#   (fixes Qwen3-VL rope index + real video timestamps; see that file's docstring)
# - +data.apply_chat_template_kwargs.* -> processor decodes the video from its
#   path: 32 frames, fps disabled, whole-video pixel budget = per-frame x 32
#   ('+' prefix: the yaml default for apply_chat_template_kwargs is an empty dict)
# - max_length=8192: measured val-set max is 4118 tokens (p95=3563) -> 2x headroom
# - LoRA r=16 on the LLM only (ViT frozen via exclude_modules), lr 1e-4
#   (LongVT full-param used 5e-5; LoRA runs one notch hotter)
#
# SFT-dose knob (plan §3 decision): SFT_DOSE=2000 for the light arm,
# unset/-1 for the full 6.1K arm.
set -xeuo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

MODEL_PATH=${MODEL_PATH:-${REPO}/models/Qwen3-VL-4B-Instruct}
EXP_NAME=${EXP_NAME:-sft_tvg_full}
SFT_DOSE=${SFT_DOSE:--1}                 # -1 = all 6.1K traces; 2000 = light arm
EPOCHS=${EPOCHS:-2}

mkdir -p ckpts logs

torchrun --standalone --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files="${REPO}/data/processed/sft_train.parquet" \
    data.val_files="${REPO}/data/processed/sft_val.parquet" \
    data.train_max_samples="${SFT_DOSE}" \
    data.custom_cls.path="${REPO}/agentic_tvg/sft_dataset.py" \
    data.custom_cls.name=TVGMultiTurnSFTDataset \
    +data.apply_chat_template_kwargs.num_frames=32 \
    +data.apply_chat_template_kwargs.fps=null \
    +data.apply_chat_template_kwargs.videos_kwargs.size.longest_edge=1605632 \
    +data.apply_chat_template_kwargs.videos_kwargs.size.shortest_edge=100352 \
    data.max_length=8192 \
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
    optim.lr_warmup_steps_ratio=0.03 \
    trainer.project_name=agentic-tvg \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${REPO}/ckpts/${EXP_NAME}" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    "$@" 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
