# Frame-Count & Config Sweep (2026-08-26)

**Host** srv1-lg2 — 1× A100 80GB PCIe, 64 cores, 188 GB RAM.
**Method** two real GRPO steps per setting (`run_grpo.sh` + overrides), TRAIN_BS=8,
K=4 unless noted; peak VRAM from 5 s `nvidia-smi` sampling, timings from verl's
`perf/` metrics.
**Question** what global frame count can SFT and RL share (DATA.md §0), and which
knobs have to move with it.

Four startup blockers were fixed before any of this could run; see
`env_setup/ENVIRONMENT.md` §8.

## 1. Token cost (measured on CPU, byte-identical to the verl path)

Global view ≈ **27.2 tok/frame** including timestamps, on a ~480 tok floor of
instruction text + tool schema.

| GLOBAL_FRAMES | prompt tokens | required MAX_PROMPT_LEN |
|---|---|---|
| 64 | 2,203 | 4096 |
| **128 (production)** | **3,944** | **4608** |
| 192 | 5,687 | 6144 |
| 256 | 7,428 | 8192 |
| 512 | ~14,500 | 16384 |

Crop-tool returns, measured at ~94 tok/frame typical and ~147 tok/frame worst
case, budgeted for 3 calls:

| CROP_NUM_FRAMES | one crop | 3 crops (worst) | required MAX_RESP_LEN | required limit_images |
|---|---|---|---|---|
| 16 | 1,524 | ~7,500 | 8192 | 64 |
| 24 | 2,268 | ~11,100 | 12288 | 80 |
| **30 (production)** | ~2.9–4.6K | **~14.7K** | **16384** | **112** |

## 2. Sweep results

All settings completed 2/2 steps except the paper config.

| Setting | MAX_PROMPT_LEN | GPU peak | train alloc/reserved | RAM peak | step time | score/mean |
|---|---|---|---|---|---|---|
| F=64 K=4 | 4096 | 39.3 GB | 29.2 / 31.3 GB | ~65 GB | 131 s | 0.32 |
| F=128 K=4 | 4608 | 38.4 GB | 30.1 / 32.3 GB | 65.6 GB | 161 s | 0.28 |
| F=192 K=4 | 6144 | 38.8 GB | 31.6 / 33.9 GB | 73.0 GB | 210 s | 0.33 |
| F=256 K=4 | 8192 | 39.0 GB | 31.0 / 33.2 GB | 76.5 GB | 242 s | 0.35 |
| F=512 K=16 (paper) | 16384 | ~39.6 GB (in rollout) | never reached | **>180 GB, OOM** | — | — |
| F=512 K=16, half batch | 16384 | 48.7 GB | never reached | 101 GB | **>30 min, aborted** | — |
| F=128 K=16 (128 traj/step) | 4608 | **55.0 GB** | 42.8 / 45.9 GB | 113.6 GB | 632 s | 0.46 / 0.28 |
| C=30 crop budget check | 4096 | — | 35.5 / 38.0 GB | 70.7 GB | passed 2/2 | 0.30 |

36 GB of every GPU peak is the vLLM reservation (`GPU_MEM_UTIL=0.45` during the
sweep; it sleeps during the training phase). Training-side activations are set by
token-packing, not by frame count — which is why **F=64 → 256 costs only ~2 GB of
VRAM** while step time nearly doubles.

## 3. Why the paper config (512 frames / K=16 / 16K generation) does not fit

1. **The videos are too short.** Our RL clips are low-frame-rate transcodes:
   total frames min 447 / median 615 / max 903. `nframes=512` exceeds some clips
   outright, and qwen-vl-utils then falls back to a `read_video` that no longer
   exists, reporting a misleading AttributeError. Capping per row fixed it (163
   of the then-current 893 rows were capped). **F ≤ 446 needs no cap at all;
   above that, `extract_rl.py` must cap per video.**
2. **Host RAM, 8 agent workers.** 128 trajectories decoding 512 full-resolution
   frames across 8 processes × 32 decode threads → 180.9 / 188 GB, Ray killed 9
   workers.
3. **Host RAM, 2 agent workers.** Throttling to `rollout.agent.num_workers=2`
   still hit 180.8 GB. The decode concurrency was never the main cost — the
   rollout *output* is: 128 trajectories × 512 frames of pixel tensor (~280 MB
   each, ~36 GB total) flows through batch assembly and Ray plasma (`/dev/shm`,
   charged to RAM) in several copies. It scales with `TRAIN_BS × K × F` and
   throttling workers cannot touch it.
4. **Wall clock, half batch.** TRAIN_BS=4 (64 trajectories) + 2 workers held RAM
   to 101 GB and VRAM to 48.7 GB, but did not finish step 1 in 30 minutes.
   The paper config *fits* on one GPU; it is the wall clock that makes it
   impossible: >30 min/step is 130+ h over the 267-step horizon. Aborted there.

## 4. Conclusions

- **The 80 GB card is not the constraint.** Everything through F=256 stayed
  ≤39 GB, and vLLM initialised even at a 32,768-token single-sequence budget.
- **The two real constraints are host RAM and wall clock**, both scaling with
  frames × trajectories per step (`TRAIN_BS × K × F`). The knobs that matter are
  `TRAIN_BS` and `rollout.agent.num_workers` — not `GPU_MEM_UTIL`.
- **F=128 is the sweet spot.** It costs +23% step time over F=64 with no memory
  pressure, and it is where the evidence-window coverage argument lands: at 64
  frames 12.4% of RL questions had <3 global frames inside the answer window; at
  128 that drops to 1.7% (median 10 in-window frames).
- **C=30 matches the paper's crop density.** LongVT's tool samples crops at 1 fps
  (measured from selftrace: median window 31 s ≈ 30 frames). A fixed 30 keeps the
  prompt schema and the response budget static, which a variable 1 fps would not.
  A C=32 variant never got past vLLM engine-core init (crash-looped across 8
  launches, one Ray OOM kill) and was abandoned rather than debugged; every
  budget in §1 assumes C=30.
- **Frame count does not move early SFT loss.** 2-step SFT smokes at
  F=96/128/160/192 all landed at val/loss 1.480–1.485, so F is chosen on
  evidence-window coverage and step-time grounds (above), not SFT fit.
- **K is free on the GPU**; it costs step time linearly and RAM through the
  trajectory count.
- Changing the frame count is not a local edit: `constants.GLOBAL_NUM_FRAMES`
  is baked into the prompt text, so SFT data must be re-rendered and
  `extract_rl.py` re-run (DATA.md §0.5), and `MAX_PROMPT_LEN` / `max_model_len` must
  move with the §1 table.

**Not the whole story:** this sweep's 2-step smokes reported 128 trajectories/step
as comfortable at 113.6 GB. It is, but the *third* step is not — host RAM ratcheted
upward until it died at step 4. That was diagnosed and fixed on 2026-08-27
(glibc allocator, `GRPO_NOTES.md` §3), and it is why the production batch below is
8 rather than the 16 this sweep suggested. A smoke has to outlive the failure
period it is meant to rule out.

## 5. Production configuration (current)

The scripts are the source of truth; this is what they say today.

```
Frames    F=128 @ ≤50,176 px/frame       constants.GLOBAL_NUM_FRAMES
Crops     C=30  @ ≤150,528 px            constants.CROP_NUM_FRAMES, 3 calls max
GRPO      K=16 × TRAIN_BS=8 = 128 trajectories/step
          267 steps = 2 epochs over 1,068 prompts, ~12.6 min/step, ~60 h
Budget    MAX_PROMPT_LEN=4608 · MAX_RESP_LEN=16384 · max_model_len=20992
Execution GPU_MEM_UTIL=0.65
          ppo / log_prob max_token_len_per_gpu = 24576
          max_num_batched_tokens=24576 · limit_images=112
          engine_kwargs.vllm.mm_processor_kwargs.max_pixels=150528
          param_offload=False · optimizer_offload=False
          MALLOC_MMAP_THRESHOLD_=131072 · MALLOC_TRIM_THRESHOLD_=134217728
SFT       GLOBAL_FRAMES=128 · data.max_length=20480 · max_token_len_per_gpu=24576
Data      SFT 600 questions · RL 1,068 train / 114 val · zero drops (DATA.md §3)
```

Two of these exist only to stop something silent from happening:
`mm_processor_kwargs.max_pixels` stops vLLM from profiling 112 images at the
preprocessor default of 16.7M px and swallowing the KV pool; the two `MALLOC_*`
variables stop the glibc heap from ratcheting (§4 above, `GRPO_NOTES.md` §3).

## 6. Reproduce

```bash
# any single setting (F=192 shown):
EXP_NAME=grpo_smoke_f192 TRAIN_FILE=data/processed/rl_train_f192.parquet \
MAX_PROMPT_LEN=6144 TRAIN_BS=8 GROUP_SIZE=4 TOTAL_STEPS=2 \
bash run_grpo.sh trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1
```

Frame variants come from rewriting `videos[].nframes` in `rl_train.parquet`.
Above F=446, cap to each video's real frame count first (§3, item 1).

## 7. Provenance

The raw smoke outputs (`results/smoke*`, `results/grpo-smoke*`, `results/qa-smoke`
— console logs and tb events, ~3.6 MB) were deleted on 2026-08-30; this file is
what survives of them. The offload A/B that ran as `grpo-smoke-opt` is written up
in `GRPO_NOTES.md` §3d (offload is not where the RAM goes). `results/memtest*`
is kept — GRPO_NOTES cites it directly. Production-run analysis lives in
`GRPO_v1_RESULTS.md`.
