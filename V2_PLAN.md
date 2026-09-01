# V2 Plan — Recipe v2: SFT2 (multi-crop injected) → GRPO2 (drafted 2026-09-01)

The user's call (2026-09-01): redo the pipeline as a clean second recipe —
**stage-1 SFT with the reflection data added, then the improved GRPO** —
instead of GRPO2_PLAN's route (rft2 on top of grpo-vanilla, then grpo2).
GRPO2_PLAN.md stays the evidence base; its §3 changes (judge v2, reward
re-budget, constant lr, shaping) all carry over unchanged. Only its §4
order of operations is superseded by this file.

## 1. Why injection moves from RFT to stage-1 SFT

- **The rft ablation point is already measured and neutral-to-weak**: v2 acc
  0.5702 vs GRPO's 0.5965, iou 0.2206 vs 0.2178, tool calls ≡ 1.0
  (`results/val-rft/analysis.md`). Stacking reflection onto that stage buys
  format unlock at best; the stage itself added nothing.
- **extract_rft.py structurally re-hardens single-crop** (its parse requires
  exactly one crop_video call, DATA.md §8). Injecting at stage 1 sidesteps
  that machinery entirely for this round.
- **The shaping term gets a whole run to work with.** With 2-crop mass in
  the prior from RL step 0, `+0.25·max(0, iou_best − iou_first)` can fire
  throughout all 267 steps. Round 1 showed what happens without support:
  the 20 early multi-crop attempts scored below group mean and GRPO
  extinguished them to 2 (GRPO2_PLAN §3d).
- **Clean v1/v2 story**: grpo-vanilla stays frozen as the v1 baseline;
  v2 = SFT2 → GRPO2 from the base model, same prompts, same val set.

Two costs, stated up front:

1. **Attribution coarsens.** GRPO2_PLAN wanted a single-axis (incentive-only)
   comparison. v2 changes data *and* incentives, so grpo2-vs-grpo-vanilla
   reads as "recipe v2 vs v1". Partial recovery: the SFT2 step-0 val
   (vs SFT1's v2 baseline 0.5395) isolates the data effect before RL touches
   anything.
2. **Round-1 RL gains are re-earned, not inherited.** GRPO2 starts from SFT2,
   not from grpo-vanilla — the ~60 h run must re-buy the ~+9 pt round-1 gain
   before showing new headroom. This is the price of the clean story.

## 2. SFT2 data build (CPU only, no GPU lock needed)

1. **Re-download geminicot** (`geminicot_1.zip` + `geminicot_2.zip`, 15.8G).
   586 train rows decode their global view from `data/videos/geminicot/*.mp4`
   at train time and those mp4s were deleted 2026-08-27 (DATA.md §5).
   Extract only the ~600 mp4s the sft parquets actually reference, then
   delete the zips (disk, §6).
2. **Fetch the reflection tier**: ~10 GiB / 70 videos / 456 traces (all
   2-crop) by byte-range from the `longvideoreflection_*.zip` entries
   (ZIP_STORED) per `data/processed/reflection_video_map.json`.
   **Correction to GRPO2_PLAN §4.2: the mp4s must STAY on disk through SFT2
   training** — the 128-frame global view is decoded from the mp4 at train
   time, not pre-rendered. Delete them after training, not after rendering.
3. **Extend `render_traces.py` to N crop windows per trace** (`render_row`
   is single-window today) and re-render the reflection traces into OUR
   template (§0.5 discipline: our prompts, canonical tool_call, frames via
   `video_frames.py`, rft_9397 placeholder guard). Prompt distribution is
   unchanged → val comparability survives.
4. **Checks before training**:
   - video-id overlap reflection × rl_val (and × selfqa) must be zero;
   - token budget: worst 2-crop trace ≈ 3.9K global + 2×4.6K crops + text
     ≈ 15K < `data.max_length=20480` — fits, no config change;
   - placeholders == assets asserted per row.
5. **Review gate (user)**: hand-read a sample of rendered reflection traces
   (rft_review precedent). Known quality caveat: the second crop's timestamp
   appears without derivation — these traces teach the retry *format*, not a
   search strategy (GRPO2_PLAN §3d-A). That is all the unlock needs.
6. **Output**: `sft2_train.parquet` = sft_train (1,958) + reflection (~456)
   ≈ 2,414 rows, ~19% 2-crop; val split by video id folded into
   `sft2_val.parquet`. `sft_train.parquet` untouched.

## 3. SFT2 train + merge + gate eval

- `SMOKE=1 TRAIN_FILES=.../sft2_train.parquet bash run_sft.sh` first, then
  the full run: `TRAIN_FILES=... VAL_FILES=... EXP_NAME=sft_mix2
  bash run_sft.sh` — from the base Qwen3-VL-4B (default MODEL_PATH),
  ~2.5 h at the measured rate.
- `python merge_adapter.py --ckpt results/sft-mix2/ckpt --out
  results/sft-mix2/merged` (base = models/Qwen3-VL-4B-Instruct).
- **Gate eval**: `MODEL_PATH=results/sft-mix2/merged bash run_grpo.sh
  trainer.val_only=True`. Two readouts decide the RL launch:
  - `num_tool_calls/mean > 1.0` — the unlock. If ≡ 1.0 after ~19% 2-crop
    SFT, escalate to the prompt nudge (GRPO2_PLAN §3d-B) before burning
    60 h; raising the reflection share to the 15 GiB / 576-trace tier is a
    ~3 h loop.
  - v2 acc vs SFT1's 0.5395 — expect ≈ or above; a drop beyond ~1 SE
    (±4.7 pts) means the reflection data hurt the QA line: investigate
    before RL.

## 4. GRPO2 — per GRPO2_PLAN §3, only the start model changes

- Implement `compute_score_qa2` (§3b: format 0/−0.5, TIME_WEIGHT=1.0,
  `+0.25·max(0, iou_best − iou_first)`); unit-test against cached
  transcripts. Judge v2 is already the live default; the three judge.py
  fixes are done.
- Launch:
  ```bash
  REWARD_FN=compute_score_qa2 EXP_NAME=grpo2 \
  MODEL_PATH=results/sft-mix2/merged \
  bash run_grpo.sh actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  ```
  TOTAL_STEPS=267, TRAIN_FILE stays `rl_train.parquet`, KL ref = SFT2
  (ref_in_actor), all RAM/plasma/glibc settings untouched.
- Monitoring, guard rails, success criteria: GRPO2_PLAN §5 verbatim
  (iou > 0.30; tool_calls > 1.2; v2 acc > 0.5965 by > 1 SE on 3-checkpoint
  means; taxonomy mechanism check via `analyze_rollouts.py`). Extra
  baseline to quote alongside: SFT2's step-0 val.
- Final numbers: VideoSIAH-Eval on grpo2's final checkpoint + a
  grpo-vanilla pass if not already done, same judge version
  (GRPO2_PLAN §4.7).

## 5. Stage 3 disposition

rft2 (RFT-level injection) is retired as the round-2 route. Stage 3 for v2,
if run at all, distills grpo2's own rollouts with `extract_rft.py`'s
single-crop hard-require removed (the fix-forward GRPO2_PLAN §3d already
demands).

## 6. Disk budget (46G free, measured 2026-09-01; no ckpt dirs on disk —
each results/* holds only its 8.3G merged model)

- Build phase: geminicot needed mp4s ~8–16G (selective extract, zips
  deleted; download one zip at a time) + reflection mp4s 10G + rendered
  frames ~1G → **~19–27G committed, ~19G free during SFT2 training**.
- The SFT checkpoint is ~19G and verl saves-before-deleting under
  `max_ckpt_to_keep=1` (2×19G transient — the known ENOSPC bite,
  run_sft.sh header). Mitigations, in order:
  1. raise `trainer.save_freq` so at most one mid-run save exists (~150
     total steps at bs 32; save_freq=75 gives one mid-run + final);
  2. delete `results/rft/merged` (8.3G) after confirming it is on the Hub
     (`hf_push.sh` / the HF account);
  3. fetch reflection in batches.
- After merge: delete `results/sft-mix2/ckpt`, keep `merged` (8.3G); delete
  reflection + geminicot mp4s before GRPO2 (RL needs only selfqa + rl_val)
  → ~40G+ free for GRPO2's 2×17G checkpoint transient.

## 7. Order of operations

1. geminicot re-download + selective extract (no GPU).
2. reflection fetch (batches) + `render_traces.py` N-crop extension +
   render + §2.4 checks (no GPU).
3. **Review gate — user reads the sample. Nothing trains before this.**
4. GPU lock per protocol, then: SFT2 smoke → SFT2 → merge → gate eval (§3).
5. `compute_score_qa2` + unit tests (can overlap with 1–2).
6. GRPO2 launch (§4), monitor per GRPO2_PLAN §5.
7. Post-run: VideoSIAH, update README/DATA/GRPO_RESULTS, stage-3 decision.
