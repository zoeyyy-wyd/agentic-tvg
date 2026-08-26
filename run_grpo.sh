#!/usr/bin/env bash
# GRPO Stage-2 (plan §5) — Qwen3-VL-4B + LoRA + crop_video multi-turn, 1x A100 80GB.
# NOTE: QA rework pending (PLAN.md 8) — the verl wiring below is verified, but
# data expects extract_rl.py output and the reward must become compute_score_qa.
#
# Every non-default key below was verified against verl 0.9.0 source:
# - rollout.mode=async + data.return_raw_chat=True    -> agent-loop path (docs/start/agentic_rl.rst)
# - agent.default_agent_loop=tool_agent               -> rows also carry agent_name="tool_agent"
# - multi_turn.tool_config_path                       -> crop_video_tool.yaml (CropVideoTool)
# - rollout.load_format=safetensors + lora_rank>0     -> LoRA-RL contract (docs/advance/ppo_lora.rst)
# - exclude_modules '.*visual.*'                      -> keep the ViT frozen; LoRA on the LLM only
# - max_user_turns=3 / max_assistant_turns=4          -> T=3 tool calls + final answer
# - limit_images=64                                   -> vLLM mm budget: 3 crops x 16 frames + slack
#
# Ablation switches (plan §6):
#   REWARD_FN=compute_score_penalty EXP_NAME=grpo_penalty bash run_grpo.sh
#   MAX_USER_TURNS=1 EXP_NAME=grpo_t1 bash run_grpo.sh                                      (multi-turn value, T=3 vs T=1)
set -xeuo pipefail
cd "$(dirname "$0")"
REPO=$(pwd)

# Guard the two silent killers before spending hours: wrong conda env, and the
# missing LD_PRELOAD that breaks torchcodec's first video decode (ENVIRONMENT.md §7).
set +x; source "${REPO}/env_setup/preflight.sh"; set -x

MODEL_PATH=${MODEL_PATH:-${REPO}/models/Qwen3-VL-4B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${REPO}/data/processed/rl_train.parquet}
VAL_FILE=${VAL_FILE:-${REPO}/data/processed/rl_val.parquet}
EXP_NAME=${EXP_NAME:-grpo_vanilla}
REWARD_FN=${REWARD_FN:-compute_score}          # compute_score | compute_score_penalty
GROUP_SIZE=${GROUP_SIZE:-8}                    # plan §5: GRPO group size
TRAIN_BS=${TRAIN_BS:-32}                       # prompts per step (plan: 32-64)
MAX_USER_TURNS=${MAX_USER_TURNS:-3}
TOTAL_STEPS=${TOTAL_STEPS:-150}                # plan: 100-200, stop at reward saturation
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.45}             # rollout side, policy shares the same GPU
LOGGER=${LOGGER:-'["console"]'}                # '["console","wandb"]' once wandb is set up

mkdir -p ckpts logs

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BS}" \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_patch_size=16 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.exclude_modules='.*visual.*' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.limit_images=64 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$((MAX_USER_TURNS + 1)) \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${REPO}/crop_video_tool.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    reward.custom_reward_function.path="${REPO}/agentic_tvg/reward.py" \
    reward.custom_reward_function.name="${REWARD_FN}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=agentic-tvg \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${REPO}/ckpts/${EXP_NAME}" \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@" 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
