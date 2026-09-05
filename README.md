# Agentic Video QA

Multi-turn tool-calling video QA: Qwen3-VL-4B + verl GRPO with a `crop_video`
tool, on one A100 80GB. Recipe adapted from LongVT (arXiv:2511.20785).

Three stages, LongVT's own shape: **SFT** cold start (their stage-3 traces
repurposed as our stage 1) → **GRPO** with judge + IoU rewards → **RFT** on our
own best rollouts (their stage 3, ours planned). One evaluator for every
stage: `bash run_grpo.sh trainer.val_only=True` — evaluation and RL share one
code path, so numbers are comparable across stages by construction.

Docs: `DATA.md` (data provenance + processing, incl. the rft_9397 bug record) ·
`GRPO_NOTES.md` (why the RL config is what it is, incl. the OOM case file and
the silent-config-key lessons) · `GRPO_v1_RESULTS.md` (round-1 forensics, incl.
the pool-saturation analysis) · `GRPO2_PLAN.md` (**the current plan**: round 2
= judge v2 + epoch-boundary curriculum + constant lr) · `V2_PLAN.md`
(superseded; post-mortem of the abandoned reflection-injection route) ·
`V2_RESULTS.md` (**the v2 results & analysis — start here for outcomes**) ·
`env_setup/ENVIRONMENT.md` (conda env `verl`) · `results/*/README.md` (runs).

## Reward

```
round 2 (compute_score_qa2, the default):  R = 0.5·format_ok + R_acc + 1.0·IoU(crop, evidence)
round 1 (compute_score_qa, frozen):        R = 0.5·format_ok + R_acc + 0.5·IoU(crop, evidence)
```

- **R_acc**: Anthropic judge, {FULL 1.0 / PARTIAL 0.5 / INCORRECT 0},
  **two frozen instruments** selected by `JUDGE_V` (recorded per run in the
  `[recipe]` console line): **v2 = the default** — claude-sonnet-5,
  question-anchored rubric, `judge_cache_v2.jsonl` (`judge_v2.py`); v1 —
  haiku, one-word rubric, `judge_cache.jsonl` (`judge.py`), reachable with
  `JUDGE_V=1` to reproduce pre-2026-09-01 numbers. v1/v2 accuracies never
  mix. Append-only caches keyed by (question, gt, answer) = deterministic,
  auditable, replays free. An API *failure* hard-stops the run
  (`JudgeUnavailable`) rather than silently falling back; deliberate offline
  mode (`JUDGE_DISABLE=1` / no key) falls back to alias matching, announces
  itself, and is used by tests only. Credits come from console.anthropic.com.
- **R_time** is fully programmatic: best IoU between any `crop_video` call
  and the evidence window; no call → 0. The 1.0 weight is measured to be
  near-inert on within-group ranking (GRPO_v1_RESULTS §4) — it decides whether
  grounding may outrank one acc tier, nothing more; no iou target rests on it.

## Layout

```
agentic_tvg/              core library (pip install -e .)
  constants.py              frame/token budget — single source of truth
  prompts.py                system/user prompt builders
  span.py  answer_match.py  answer parsing, temporal IoU, GT alias expansion
  judge.py  judge_v2.py     R_acc instruments v1 (haiku) / v2 (sonnet, default)
  reward.py                 verl reward functions (qa = round 1, qa2 = round 2)
  video_frames.py           PyAV interval sampling (shared: tool + render)
  crop_video_tool.py        verl BaseTool (the model-callable tool)
  sft_dataset.py            Qwen3-VL fixes over verl's MultiTurnSFTDataset
prepare_data.sh           downloads + renders all training data
data_prep/                render_traces.py (SFT) · extract_rl.py (RL)
run_sft.sh                SFT (SMOKE=1 for a 2-step smoke)
merge_adapter.py          SFT checkpoint -> merged HF model (GRPO's init AND ref)
run_grpo.sh               GRPO (trainer.val_only=True turns it into the evaluator)
plot_sft.py plot_grpo.py  curves.png + metrics.csv (plot_grpo also reads
                          results/<run>/tb dirs, which survive crashes)
diagnose_shm.sh           who is holding /dev/shm, live (OOM case file's tool)
results/<run>/            ckpt/ + curves + metrics + config snapshot per run
tests/  env_setup/
```

## State (2026-09-01)

| Stage | Status |
|---|---|
| Data | 1,958 SFT rows · 1,068 RL train · 114 RL val (`DATA.md`) — **regenerate after the 2026-09-01 machine wipe** (`prepare_data.sh`; only the committed repo survived) |
| SFT | **done** — val/loss 1.124 → 0.938, ~2 h; `results/sft-mix/merged` being restored from the Hub |
| GRPO round 1 | **done** — `grpo-vanilla`, 267 steps: v2 acc 0.456 → **0.5965**, iou plateaued ~0.21, tool calls ≡ 1; plateau cause = pool saturation (GRPO_v1_RESULTS §4) |
| RFT round 1 | **done and neutral** — v2 acc 0.5702 vs GRPO's 0.5965, iou 0.2206, tool calls ≡ 1 (`results/val-rft/analysis.md`); stage 3 earns nothing on round-1 rollouts |
| GRPO round 2 | **done** (2026-09-04) — `grpo_v2`: closing 3-ckpt mean acc **0.6214** (+1.9 vs v1's 0.6023), iou 0.254 breaking v1's 0.21 plateau, in 237 steps vs v1's 267. **Final model: `results/grpo-v2/merged`** |
| RFT round 2 | **done, neutral ×5** — best variant (paper dual criterion, lr 2e-5) ties grpo_v2; see V2_RESULTS §5 for the LongVT reconciliation |
| External probe | Charades-STA zero-shot grounding: SFT→GRPO **+15.1 pt R@0.5** (6 SE, n=399) — the iou gain transfers out of domain |

Val accuracies above are the v2 judge scale (SFT 0.5395 · GRPO 0.5965 ·
RFT 0.5702); the v1-scale history (0.4561/0.5044/0.5044) is frozen with
`JUDGE_V=1`.

**Baseline vs SFT, n=114 paired** (same rows, greedy, **judge v1 scale**; SFT
arm = the GRPO run's step-0 validation, i.e. exactly the model RL starts from):

| | base | SFT | Δ |
|---|---:|---:|---:|
| format_score | 0.0000 | 0.4956 | +0.4956 |
| answered | 0.3070 | 0.9912 | +0.6842 |
| acc | 0.1447 | 0.4518 | **+0.3071** |
| evidence_iou | 0.1193 | 0.1411 | +0.0218 |
| num_tool_calls | 1.9298 | 0.9912 | −0.9386 |
| reward | 0.2044 | 1.0179 | +0.8135 |

Reading: the base model already *called* the tool fine (1.93/row); it never
emitted a parseable `<think>/<answer>`, so only 31% of rows answered. SFT
bought the output form *and* 3.1× accuracy. IoU barely moved — that headroom
(0.14 of 0.5) is what GRPO's R_time term is for. Per-row dump:
`results/grpo-vanilla/val_rollouts/0.jsonl`. Two cautions, both learned the
hard way: per-row results are NOT transferable between merges of the same
checkpoint (a float-precision change flipped 104/114 greedy trajectories while
aggregates moved <0.02), and an n=10 pilot mis-read the acc gain as
formatting-only — small slices lie.

## RFT (stage 3)

Built and run once on round-1 rollouts (`data_prep/extract_rft.py`, funnel and
review record in DATA.md §8) — **outcome neutral** (state table above), so
stage 3 is not part of the round-2 critical path. If grpo_v2's rollouts look
worth distilling, the tooling already points at them: `extract_rft.py`
defaults to `results/grpo-v2/rollouts` with output prefix `rft_v2`, and
`run_rft.sh` trains from `results/grpo-v2/merged` into `results/rft-v2/`.
Keep (rename) `rollouts/` from a finished run before any rerun under the same
EXP_NAME — files are overwritten one step at a time.

## Run

```bash
bash prepare_data.sh                     # data: ~31G download + render
SMOKE=1 bash run_sft.sh                  # 2-step smoke (~7 min), then rm -rf results/smoke
bash run_sft.sh                          # SFT, ~2 h -> results/sft-mix/
python merge_adapter.py                  # fold LoRA -> results/sft-mix/merged

# GRPO round 2 (defaults: grpo_v2, compute_score_qa2, JUDGE_V=2, constant lr;
# two-stage curriculum per GRPO2_PLAN §3e/§4):
EPOCHS=1 bash run_grpo.sh                                  # stage 1, 133 steps ~30 h
python data_prep/filter_mastered.py --rollouts results/grpo-v2/rollouts   # drop mastered prompts (visit acc >= 0.9)
bash run_grpo_stage2.sh                  # stage 2 (~106 steps ~22 h); derives EPOCHS/TOTAL_STEPS + data.pt surgery itself

bash run_grpo.sh trainer.val_only=True   # the evaluator (any stage, any model via MODEL_PATH=)
python data_prep/extract_rft.py          # RFT set from grpo-v2 rollouts (prefix rft_v2)
bash run_rft.sh                          # RFT, ~3 h -> results/rft-v2/ (wraps run_sft.sh; SMOKE=1 works)
bash replot.sh                           # curves any time (plot_grpo.py on results/grpo-v2/tb)
```

Round-1 reproduction: `JUDGE_V=1 REWARD_FN=compute_score_qa
EXP_NAME=grpo_vanilla bash run_grpo.sh
actor_rollout_ref.actor.optim.lr_scheduler_type=cosine` — all four together
or the numbers are not comparable to the frozen round-1 results.

During GRPO, two things need a human:

- **Delete the superseded checkpoint after every save** (20 steps ≈ 4.5 h):
  `ls results/grpo-v2/ckpt/` must show exactly one `global_step_*` — keep
  the one in `latest_checkpointed_iteration.txt`. Each is ~17G and a save
  writes new-before-deleting-old. (`max_actor_ckpt_to_keep=1` automates this
  for fresh runs, but a resumed process never deletes the checkpoint it
  resumed *from*.)
- **Watch val acc against train reward** — train up while val flat is reward
  hacking; `response_length/mean` leaving its ~3.2K baseline is the usual
  mechanism.

To resume after an interruption **within a stage**: rerun the stage's exact
launch line (stage 2 included — its `TOTAL_STEPS`/`TRAIN_FILE` must be
repeated; the `[recipe]` line in console.log has what the stage was launched
with). With constant lr the schedule no longer bends on resume; the only
resume surgery is the documented `data.pt` move at the stage-1→2 boundary.
Run in tmux (`tmux new -s grpo`); every new terminal needs
`conda activate verl`.
