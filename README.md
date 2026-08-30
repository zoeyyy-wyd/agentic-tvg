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
the silent-config-key lessons) · `env_setup/ENVIRONMENT.md` (conda env `verl`,
incl. the patched-verl and hosts-file records) · `results/*/README.md` (runs).

## Reward

```
R = 0.5·format_ok + R_acc + 0.5·IoU(crop window, evidence window)
```

- **R_acc**: Anthropic judge (claude-haiku-4-5), LongVT's own rubric —
  FULL 1.0 / PARTIAL 0.5 / INCORRECT 0. Temperature 0 + append-only cache
  (`data/processed/judge_cache.jsonl`) = deterministic, auditable, replays
  free. An API *failure* hard-stops the run (`JudgeUnavailable`) rather than
  silently falling back — a mid-run scorer swap would corrupt training.
  Deliberate offline mode (`JUDGE_DISABLE=1` / no key) falls back to alias
  matching and announces itself; used by tests only, not comparable to judged
  runs. Credits come from console.anthropic.com — claude.ai usage credits are
  a different pool with the same error text. ~$0.3/step at 128 trajectories.
- **R_time** is fully programmatic: best IoU between any `crop_video` call
  and the evidence window; no call → 0.

## Layout

```
agentic_tvg/              core library (pip install -e .)
  constants.py              frame/token budget — single source of truth
  prompts.py                system/user prompt builders
  span.py  answer_match.py  answer parsing, temporal IoU, GT alias expansion
  judge.py  reward.py       LongVT-rubric judge + the verl reward functions
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

## State (2026-08-28)

| Stage | Status |
|---|---|
| Data | 1,958 SFT rows · 1,068 RL train · 114 RL val (`DATA.md`) |
| SFT | **done** — val/loss 1.124 → 0.938, ~2 h, `results/sft-mix/` |
| GRPO | **67/267** — val acc 0.452 → ~0.50, no hacking signature; 4 OOMs and one judge-credit outage survived (GRPO_NOTES §6) |
| RFT | planned — see recipe below |

**Baseline vs SFT, n=114 paired** (same rows, greedy, judge both arms; SFT arm
= the GRPO run's step-0 validation, i.e. exactly the model RL starts from):

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

## RFT (stage 3) recipe

Raw material accumulates for free: `results/grpo-vanilla/rollouts/<step>.jsonl`
holds every trajectory with full text + reward breakdown. For a future
`data_prep/extract_rft.py`:

1. filter `score > 1.5` (~27% of trajectories ⇒ ~9K over a full run);
2. dedupe per question, ≤3 answer-distinct traces (as
   `render_traces.pick_traces`) — K=16 × 2 epochs gives up to 32 candidates
   per question, and taking all lets easy prompts flood the set;
3. rebuild `<image>` placeholders + crop frames from the logged `crop_video`
   windows via `video_frames.py` — dumps store tool responses as text only;
   same code path as the RL tool, zero train/serve skew;
4. prefer late-stage rollouts, and **hand-read a sample of `score > 1.8` rows
   first**: RFT bakes reward quirks into weights harder than RL — no KL pulls
   back. Trace-back is verified: question → `rl_train.parquet` is unique, and
   the crop window parses out of the logged tool call.

Then SFT on the merged GRPO model with the same `run_sft.sh` machinery, and
re-evaluate. Keep (rename) `rollouts/` from the finished run first — a rerun
with the same EXP_NAME overwrites `<step>.jsonl` one file at a time.

## Run

```bash
bash prepare_data.sh                     # data: ~31G download + render
SMOKE=1 bash run_sft.sh                  # 2-step smoke (~7 min), then rm -rf results/smoke
bash run_sft.sh                          # SFT, ~2 h -> results/sft-mix/
python merge_adapter.py                  # fold LoRA -> results/sft-mix/merged
bash run_grpo.sh                         # GRPO, ~58 h -> results/grpo-vanilla/
bash run_grpo.sh trainer.val_only=True   # the evaluator (any stage, any model via MODEL_PATH=)
bash replot.sh                                       # curves any time (runs plot_grpo.py on results/grpo-vanilla/tb)
```

During GRPO, two things need a human:

- **Delete the superseded checkpoint after every save** (20 steps ≈ 4.5 h):
  `ls results/grpo-vanilla/ckpt/` must show exactly one `global_step_*` — keep
  the one in `latest_checkpointed_iteration.txt`. Each is ~17G and a save
  writes new-before-deleting-old. (`max_actor_ckpt_to_keep=1` automates this
  for fresh runs, but a resumed process never deletes the checkpoint it
  resumed *from*.)
- **Watch val acc against train reward** — train up while val flat is reward
  hacking; `response_length/mean` leaving its ~3.2K baseline is the usual
  mechanism.

To resume after any interruption: `bash run_grpo.sh`, **no arguments** —
setting `TOTAL_STEPS` on a resume reshapes the cosine schedule and the lr
jumps. Run in tmux (`tmux new -s grpo`); every new terminal needs
`conda activate verl`.
