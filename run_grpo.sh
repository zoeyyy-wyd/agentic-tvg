#!/usr/bin/env bash
# GRPO Stage-2 (plan §5) — Qwen3-VL-4B + LoRA + crop_video multi-turn, 1x A100 80GB.
# QA-reworked 2026-08-26: data from extract_rl.py, reward = compute_score_qa.
#
# Every non-default key below was verified against verl 0.9.0 source:
# - rollout.mode=async + data.return_raw_chat=True    -> agent-loop path (docs/start/agentic_rl.rst)
# - agent.default_agent_loop=tool_agent               -> rows also carry agent_name="tool_agent"
# - multi_turn.tool_config_path                       -> crop_video_tool.yaml (CropVideoTool)
# - rollout.load_format=safetensors + lora_rank>0     -> LoRA-RL contract (docs/advance/ppo_lora.rst)
# - fsdp_config.*_offload=False                       -> a SPEED choice, not a memory fix.
#   param_offload exists for full-parameter finetuning, where params+grads+Adam
#   states cannot coexist with vLLM. With LoRA only the adapter is trainable
#   (126 MiB of weights, 265 MiB of optimizer state), so resident need is ~18G
#   against the 80-52=28G that survives vLLM waking up. Measured 2026-08-27:
#   785 s/step vs 876 s with offload on -- 91 s/step, ~6.7 h over 267 steps,
#   from not shuttling 35 G of params across PCIe twice per step.
#   It barely moves host RAM (peak 171 vs 175 G): the CPU-side WorkerDict
#   footprint is FSDP scaffolding, not the parked params. Cost is VRAM
#   headroom, 23 G -> 6.5 G, spent at the weight-sync instant (73.5/80 G,
#   identical on all 4 measured steps -- deterministic, not load-dependent).
#   If that is too thin: gpu_memory_utilization=0.55 buys back ~8 G.
# - actor.use_kl_loss=True (NOT algorithm.use_kl_in_reward) -> the GRPO paper puts
#   KL in the loss, not in the reward; mixing it into the reward would also
#   corrupt the reward numbers we report. Free here: lora_rank>0 makes verl set
#   ref_in_actor (ray_trainer.py:360), so the reference policy is this same
#   actor with the adapter switched off -- no second worker, no extra VRAM.
# - exclude_modules '.*visual.*'                      -> keep the ViT frozen; LoRA on the LLM only
# - max_user_turns=3 / max_assistant_turns=4          -> T=3 tool calls + final answer
# - limit_images=112                                  -> vLLM mm budget: 3 crops x 30 frames + slack
# - engine_kwargs.vllm.mm_processor_kwargs            -> cap profiling dummy images at the real crop
#   .max_pixels=150528                                   size; without it vLLM profiles 112 images at
#                                                        the preprocessor default 16.7M px and eats
#                                                        the whole KV pool (FRAMES_SWEEP.md §3.5/§4.5)
# - data.val_batch_size=2                             -> loader batch, NOT a row cap:
#   all 114 val rows still run, two at a time. The val at test_freq=20 fires with
#   the training state resident (actor + optimizer + vLLM), which is where the
#   2026-08-26 c30 smoke lost a DataLoader worker to `signal: Killed`; unset, the
#   loader takes all 114 in one bite. A val_only pass over 114 is fine on its own
#   (measured the same day), so this is about the in-training slot.
#   Deliberately NOT capping rows with val_max_samples: it would buy ~1.2h of 52h
#   and silently make every val_only run a subset of the val set.
# - trainer.use_v1=False                              -> V1 TaskRunner imports `transfer_queue`,
#                                                       which the verl 0.9.0 wheel neither ships nor
#                                                       declares as a dep (ENVIRONMENT.md §8)
# - rollout.max_model_len=prompt+response             -> left unset it falls back to
#                                                       max_position_embeddings=262144, whose single-
#                                                       sequence KV (~36G) exceeds the 0.45-util KV
#                                                       pool -> vLLM refuses to init (ENVIRONMENT.md §8)
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

# GRPO starts from the SFT-merged model, not the raw base -- pointing at the
# base silently discards the cold start, and you would not find out until the
# run ends ~52h later. Made the default rather than an env var you must
# remember; the zero-shot baseline is the exception and states itself:
#   MODEL_PATH=${REPO}/models/Qwen3-VL-4B-Instruct bash run_grpo.sh ...
MODEL_PATH=${MODEL_PATH:-${REPO}/results/sft-mix/merged}
[ -d "${MODEL_PATH}" ] || { echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "  build it with: python export_adapter.py" >&2
    echo "  or pass MODEL_PATH=... explicitly (e.g. the base model for a baseline)" >&2
    exit 1; }
TRAIN_FILE=${TRAIN_FILE:-${REPO}/data/processed/rl_train.parquet}
VAL_FILE=${VAL_FILE:-${REPO}/data/processed/rl_val.parquet}
# Token budget knobs (frames sweep 2026-08-26: prompt ≈ 27 tok/frame + ~480
# text/schema, so F=64→2.2K, F=128→3.9K, F=192→5.7K, F=256→7.4K; raise
# MAX_PROMPT_LEN together with the parquet's nframes).
# Production values (FRAMES_SWEEP.md §4.5): F=128 → prompt 4608; C=30 crops
# (3 × ~4.6K worst) + reasoning → response 16384.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4608}
MAX_RESP_LEN=${MAX_RESP_LEN:-16384}
EXP_NAME=${EXP_NAME:-grpo_vanilla}
REWARD_FN=${REWARD_FN:-compute_score_qa}       # PLAN.md 5: format + alias match + evidence IoU
GROUP_SIZE=${GROUP_SIZE:-16}                   # K=16 (FRAMES_SWEEP §4.5; GPU-free, costs wall time)
# prompts/step. 8 x K=16 = 128 trajectories/step. Was 16 (=256 traj) until
# 2026-08-27, when step 1 died with ray OutOfMemoryError at the vLLM weight
# sync: the node has 188G of RAM and TaskRunner alone held 95.8G, because every
# in-flight trajectory carries its decoded video frames as CPU tensors. K stays
# at 16 -- GRPO's advantage is a within-group comparison, so K is the part that
# must not shrink; batch only controls how many distinct prompts per step.
# Measured at 8x16 by a 3-step smoke the same day: peak RSS 132G/188G (56G of
# headroom) and 13.6 min/step. Halving the batch did NOT halve step time --
# weight sync and the vLLM sleep/wake cycle are fixed costs per step.
TRAIN_BS=${TRAIN_BS:-8}

# lr: cosine decay from 1e-5 to 0.1x over TOTAL_STEPS (verl default is
# `constant`; RL usually keeps it flat because the policy -- and therefore the
# objective -- moves under the optimizer, so "anneal toward a fixed optimum"
# does not strictly apply). Two things follow from choosing cosine here:
#   - TOTAL_STEPS is now part of the schedule, not just a stopping point.
#     Changing it reshapes the whole curve, and resuming a run under a
#     different TOTAL_STEPS makes the lr jump rather than continue.
#   - min_lr_ratio=0.1 keeps a floor. At the default 0.0 the lr reaches
#     exactly 0 at the last step, so a run that goes the distance spends its
#     final steps not learning -- and this one is budgeted at ~52h, long
#     enough that being cut short is the likely outcome.
MAX_USER_TURNS=${MAX_USER_TURNS:-3}
# 267 steps = 2.00 epochs over the 1,068 rl_train prompts (batch 8), 34.2K
# trajectories, ~60h at the measured 13.6 min/step. Sized long on purpose: the
# asymmetry favours it. Overshooting costs only the marginal hours -- stop early
# and min_lr_ratio=0.1 means lr never decayed to nothing -- while undershooting
# cannot be extended (see below) and would mean re-running from the SFT init.
# Headroom says there is something to find: at the SFT start the format term is
# already 0.487/0.5 but evidence_iou is 0.075/0.5, and the 3-step smoke moved
# reward 0.998 -> 1.165, so this is not a policy that saturates immediately.
# Since lr became cosine this is a *horizon*, not a cap:
# it is the denominator of the anneal, so plan to run it out. Two consequences:
# stopping early at reward saturation leaves the anneal unfinished, and
# resuming with a different TOTAL_STEPS makes lr jump rather than continue
# (the scheduler checkpoints its step counter, but the curve is a closure
# rebuilt from this value -- torch_functional.py, get_cosine_schedule_with_warmup).
# To resume, change nothing: `bash run_grpo.sh` and resume_mode=auto does it.
TOTAL_STEPS=${TOTAL_STEPS:-267}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}             # 0.65 doubles KV pool vs 0.45; actor offloads during rollout
LOGGER=${LOGGER:-'["console","tensorboard"]'}

# results/<name>/ is the deliverable, same convention as run_sft.sh: ckpt/ +
# rollouts/ + the config snapshot in one place. Hyphens here, underscores in
# EXP_NAME (log and tensorboard filenames already use them).
RESULT_NAME=${RESULT_NAME:-${EXP_NAME//_/-}}
RESULT_DIR=${REPO}/results/${RESULT_NAME}

mkdir -p "${RESULT_DIR}" logs
export TENSORBOARD_DIR="${REPO}/logs/tb/${EXP_NAME}"

# glibc 分配器：这两个是 2026-08-27 三次 CPU OOM 的解药，别删。
# glibc 的 mmap 门槛是动态的——每释放一个大块就上调到那个块的大小（上限
# 32MB）且只升不降。本项目的分配尺寸（单帧 588KB、crop 52MB）正落在这个区
# 间，门槛一旦被顶到上限，后续分配改从堆里拿，free 后还不给内核，RSS 只涨
# 不跌：实测 111 → 112 → 129 → 崩（182/188）。显式设置会关掉动态调整。
# 加上之后 8 步实测：96→99→116→132→99→100→100→101，涨上去能回落。
# 注意结尾的下划线是 glibc 的命名规范，漏了不生效且不报错。
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-131072}
export MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-134217728}

# rollout_data_dir vs validation_data_dir: not interchangeable. The first is
# read only inside the training loop (ray_trainer.py:1697); _validate dumps to
# the second (:696), so a val_only run with just the first writes nothing.
#
# NB: never put a `#` comment between the backslash-continued lines of the
# python3 invocation below. bash joins the lines first, so the comment then
# swallows the REST OF THE COMMAND -- including the trailing "$@". That
# silently dropped val_only/val_max_samples on 2026-08-26 and turned a 10-row
# eval into a full 114-row validation followed by real GRPO training.
#
# Do NOT export PYTORCH_CUDA_ALLOC_CONF=expandable_segments here (run_sft.sh
# does, but SFT has no vLLM): verl toggles it at runtime itself -- ON during
# training phases (fragmentation control), OFF around vLLM wake/weight-sync
# (sleep-mode CuMemAllocator conflict). A global export would leak into the
# vLLM server process and hit exactly that conflict. engine_workers.py:760/805.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BS}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESP_LEN}" \
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
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BS}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESP_LEN)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    +actor_rollout_ref.rollout.limit_images=112 \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs={max_pixels:150528}' \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=24576 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
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
    trainer.use_v1=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger="${LOGGER}" \
    trainer.project_name=agentic-tvg \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${RESULT_DIR}/ckpt" \
    trainer.rollout_data_dir="${RESULT_DIR}/rollouts" \
    trainer.validation_data_dir="${RESULT_DIR}/val_rollouts" \
    data.val_batch_size=2 \
    trainer.val_before_train=True \
    +trainer.max_ckpt_to_keep=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@" 2>&1 | tee "logs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"
