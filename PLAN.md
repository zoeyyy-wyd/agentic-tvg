# Agentic Video QA — Project Plan

**Multi-turn tool-calling video QA with verifiable rewards: Qwen3-VL-4B + verl
GRPO on one A100 80GB**

v2.1 | 2026-08-26 | Status: data pipeline built and verified; SFT ready to launch.

Supersedes the original pure-TVG plan (v1.0). The TVG line was removed
entirely on 2026-08-26; git history keeps its artifacts. Every number in
this document was measured against the released data on 2026-08-25/26; nothing
is transcribed from the paper.

---

## 1. Positioning

Port the long-video agentic QA pipeline of LongVT (arXiv:2511.20785) to
Qwen3-VL, replacing its single non-verifiable component — LLM-as-a-Judge
answer scoring — with fully programmatic rewards: alias-set containment
matching for the answer plus evidence-window IoU for the tool call. This keeps
the clean single-GPU RLVR setup that motivated the project.

Relation to the original TVG plan: TVG dropped QA because there the tool's
return value and the answer are the same kind of object (a time interval), so
the agentic loop degenerates into iterative refinement. QA restores the tool's
evidential role — `crop_video` fetches evidence, the answer interprets it —
while answer-length stratification keeps the reward verifiable.

## 2. Goal (all ablations cut on 2026-08-26 for time)

One question: **does the LongVT recipe, with the judge replaced by verifiable
rewards, hold up on Qwen3-VL-4B on a single GPU?**

Pipeline: SFT (~2K tool traces) → GRPO (one configuration) → evaluation on
rl_val, reported relative to the zero-shot baseline.

Former ablation points, now fixed in place: cold-start dose = ~2K; the reward
includes R_time (λ>0, following the paper's own recipe); max turns T=3.

Two retained items are *not* ablations: the zero-shot baseline (the anchor
that "relative improvement" is measured against) and the matcher audit gate
(§5, launch QC for GRPO). The cut arms — dose 0 / general-CoT mix, λ=0, T=1 —
are future work.

## 3. Data (all numbers measured; full per-file provenance in `DATA.md`)

### 3.1 Lineage

selftrace (15,354 traces) is LongVT's stage-3 RFT data: its model's successful
RL rollouts (answer judged correct AND crop-vs-evidence IoU ≥ 0.3) over the
selfqa questions. Deduplicated, it holds 1,290 unique questions with a median
of 7 redundant solutions each. Joined against selfqa's 1,668 questions on
exact question text: 1,157 match; 132 are selftrace-only (their videos ship
only in a 51.5 GiB archive we do not buy — dropped); 510 selfqa questions have
no trace at all — the questions their model never solved.

### 3.2 Allocation by answer verifiability (normalized GT ≤ 6 words)

|                     | verifiable                 | unverifiable        |
|---------------------|----------------------------|---------------------|
| with traces (1,157) | 918: 360 → SFT, 558 → RL   | 240 → all to SFT    |
| no traces (510)     | 335: all → RL              | 175 → dropped       |

SFT questions that RL's matcher could never score go to SFT (imitation does
not need a checker); the RL pool keeps only match-scorable answers. The 360
are sampled with weight 1/n_traces to lean toward the questions their model
rarely solved.

### 3.3 The three sets

| Set  | Composition | Size |
|------|-------------|------|
| SFT  | 600 selftrace questions × ≤3 answer-distinct traces = 1,379 rows, + 600 geminicot rows (question diversity; the paper's genuine stage-1 QA data) | 1,958 rows → `sft_train` (1,923) / `sft_val` (35), 2% val split by video id |
| RL   | remaining selfqa questions, question/video-disjoint from SFT | 1,068 → `rl_train.parquet` (was 893 in the matcher era; the LLM judge scores long answers, so the length cut is gone) |
| Eval | rl_val, verbatim | 114 (zero video overlap with selftrace — verified) |

The row ratio selftrace : geminicot ≈ 7 : 3 (question ratio 1 : 1). Knobs in
`data_prep/render_traces.py`: `--sft-questions / --traces-per-q /
--geminicot-n / --max-gt-words`.

### 3.4 Disclosed biases

- **SFT easy-question bias**: traces exist only where their model succeeded
  (inherent to distillation); partially offset by the 1/n_traces sampling.
- **RL hard-question bias**: the 335 no-trace questions are their model's
  total failures — good (that is where RL has headroom), but broken GT hides
  among them; the difficulty filter removes those as all-wrong groups, which
  is why the RL pool is capped at 1,068.
- **Cold-start concentration confound**: our cold start is doubly-filtered
  successful trajectories — "stronger" per sample than the paper's stage-1
  mixture, so RL's marginal gain will read smaller than theirs.
- **Trace parentage**: selftrace and geminicot are Qwen2.5-VL and Gemini
  output distributions respectively. If the zero-shot probe shows a high
  native success rate, best-of-N self-bootstrapping could replace
  cross-model distillation entirely (future work).

## 4. Re-rendering discipline

Principle: **cold start is calibration to the RL environment. Every byte the
model will *see* at RL time must be generated by the RL-time code; every byte
it must *produce* must be in the exact surface form the RL-time parser
accepts. The source traces contribute content only.**

Implemented in `data_prep/render_traces.py` (dual-source: selftrace by
default, `--geminicot-n` mixes geminicot):

1. system/user prompts rebuilt from `agentic_tvg/prompts.py` (QA mode). The
   upstream sentence "The Video path for this video is: X.mp4" is removed —
   it teaches the model to echo a parameter our tool schema does not have.
2. tool_call canonicalized: byte-identical to the Qwen3 chat template's own
   serialization; the `video_path` argument dropped.
3. Tool-response frames re-decoded from the mp4s: the trace's own crop window
   goes through `agentic_tvg/video_frames.py` — the same function the RL tool
   executes — for zero train/serve skew. (Also saves the 51.5 GiB archive
   that holds their pre-cropped jpgs.)
4. train/val split by video id: each question carries several traces; a
   row-level split would leak solved questions into val.
5. `<image>`/`<video>` scrubbed out of the copied think/answer text. verl's
   SFT dataset splits *every* message string on those two tokens regardless of
   role, so a literal one inside a trace's reasoning is consumed as a real
   image placeholder and shifts the whole `images` list off by one. Exactly 1
   of 15,354 selftrace traces carries one (`rft_9397`, a model-echoed tool
   header); it killed a full SFT run at step 50/60 on 2026-08-26, 57 minutes
   in, with `IndexError: list index out of range` in a dataloader worker.
   `render_traces.py` now scrubs at parse time and asserts
   placeholders == assets for every row before writing the parquet, so the
   same defect fails in 2 minutes instead of an hour.

## 5. Reward (revised 2026-08-26: paper-rubric judge over a verifiable fast path)

```
R = 0.5·format_ok + R_acc + λ·IoU(crop window, video_segment)      λ = 0.5
```

**R_acc = the judge, one instrument** (revised again 2026-08-26: the free
matcher fast-path was removed — with caching it saved only ~$1-3/run and
created a matcher-vs-judge grading seam):

- Every parsed answer goes to the **Anthropic-API judge** (default
  claude-haiku-4-5), graded with LongVT's own rubric — FULL 1.0 / PARTIAL 0.5
  / INCORRECT 0; enumerations and hedges are instructed INCORRECT.
- **Determinism & audit**: temperature 0 + append-only cache
  (data/processed/judge_cache.jsonl); one verdict per unique
  (question, GT, normalized answer) triple, ever. The cache is the audit
  trail, and repeats cost nothing.
- **Fail-safe / offline mode**: no ANTHROPIC_API_KEY, JUDGE_DISABLE=1, or API
  failure → binary fallback: rule-alias containment computed on the fly
  (answer_match.py) — deterministic, no network, used by smokes and tests.
- ground_truth in the RL parquet is the **raw GT text** (no baked aliases; no
  enrichment step — the judge covers semantics).
- **R_time** stays fully programmatic: best IoU between any crop_video call
  and the evidence window; no tool call → 0.
- Semantic alias enrichment is superseded by the judge (the extract_rl.py
  enrichment hook remains for optional frozen additions).

The zero-shot baseline and the audit gate remain: the audit now measures the
*combined* tier-1+tier-2 scorer against a stronger offline check (and human
spot-checks of the judge cache).

## 6. Training configuration

The scripts are the source of truth; these are the values as of 2026-08-27
and why they are what they are.

- **Global view = 128 frames** (`constants.GLOBAL_NUM_FRAMES`, raised from 64
  on 2026-08-26 — coverage evidence in `FRAMES_SWEEP.md`). Baked into the
  system prompt, so SFT and RL must share it and changing it means re-rendering.
- **SFT** — `run_sft.sh`: LoRA r=16 on the LLM only (ViT frozen via
  `exclude_modules`), lr 1e-4 constant with 10% warmup, 2 epochs = 120 steps,
  `max_length=20480`, ~2 h. Ran 2026-08-26; results in `results/sft-mix/`.
- **GRPO** — `run_grpo.sh`: batch 8 x K=16 = 128 trajectories/step, 267 steps
  (2 epochs), cosine lr 1e-5 -> 1e-6, KL-in-loss 0.001. 13.6 min/step ~= 60 h,
  and one judge call per trajectory ~= 34K over the run.
  Batch was 16 until step 1 died with a CPU OOM: at 256 trajectories the
  in-flight video tensors alone held 95.8 G of the node's 188 G. At 128 the
  measured peak is 132 G.

## 7. Honest gaps vs LongVT

| Dimension | LongVT | This project | Nature |
|---|---|---|---|
| SFT | 247.9K samples, 64 GPUs, full-param | ~2K, LoRA, 1 GPU | 125× gap — the bet that a strong instruct base carries the general ability itself |
| RL prompts | 1.6K | 1,068 | ~1.5× gap |
| RL config | K=16, 16K new tokens, 36K prompt | K=8, 12K total budget | single-GPU constraint |
| R_acc | LLM-as-a-Judge {1, 0.5, 0} | alias containment matcher (drops the 25% long-answer questions) | **the core methodological claim** |
| RFT | 15.4K own-rollout traces | their RFT data repurposed as our cold start; our own stage 3 pending | deliberate, disclosed |
| Base | Qwen2.5-VL-7B | Qwen3-VL-4B | generation change |

## 8. Remaining work

- [x] `data_prep/extract_rl.py` — 1,068 + 114 rows, zero drops
- [x] `agentic_tvg/answer_match.py` — word-level containment + length cap (31 tests)
- [x] `agentic_tvg/reward.py::compute_score_qa` — 0.5·fmt + acc + 0.5·evidence-IoU
- [x] eval — no separate script or serving needed: GRPO's own `val_only` path
      *is* the eval, so evaluation and RL cannot drift apart (§4).
      `trainer.validation_data_dir` writes the per-row answer/GT dump the audit
      gate wanted; `rollout_data_dir` is the training-loop key and writes
      nothing under `val_only`. Both dump `<step>.jsonl`, so a run accumulates
      its whole history rather than overwriting.
- [x] eval results, n=114 paired, greedy, judge in both arms:

      | | base | SFT | Δ |
      |---|---:|---:|---:|
      | format_score | 0.0000 | 0.4956 | +0.4956 |
      | answered | 0.3070 | 0.9912 | +0.6842 |
      | acc | 0.1447 | 0.4518 | **+0.3071** |
      | evidence_iou | 0.1193 | 0.1411 | +0.0218 |
      | num_tool_calls | 1.9298 | 0.9912 | −0.9386 |
      | reward | 0.2044 | 1.0179 | +0.8135 |

      SFT arm = step-0 validation of the GRPO run, so it is measured on exactly
      the model RL starts from; per-row dump at
      `results/grpo-vanilla/val_rollouts/0.jsonl`. The base arm was run
      2026-08-26 22:47 and has aggregate numbers only — no dump, no file.

      Reading: the base model was never bad at *calling* the tool (1.93
      calls/row, IoU 0.119), it was bad at stopping — it never emitted a
      parseable `<think>/<answer>`, so only 31% of rows answered at all. SFT
      bought the output form *and* real accuracy (3.1x). IoU barely moved,
      which is the RL R_time term's job. An n=10 pilot the day before put the
      acc delta at +0.05 and concluded the gain was formatting only; that was
      a small-sample artifact.

      Caution: the two arms must come from one merge. Re-merging the same SFT
      checkpoint via PEFT instead of a hand-rolled fold changed only float
      precision, yet under greedy decoding 104 of 114 trajectories diverged —
      one flipped argmax and the rest of the sequence follows. Aggregates
      moved <0.02, but per-row results are not transferable between merges.
- [x] GRPO script rework — `run_grpo.sh` header records every key verified
      against verl 0.9.0 source
- [x] GRPO validation smoke — first `val-core`/`val-aux` metrics this repo has
      produced. `bash run_grpo.sh trainer.val_only=True` (returns straight
      after `_validate()`, ray_trainer.py:1418). `val_batch_size` is the loader
      batch, NOT a row cap.
- [x] GRPO memory smoke — 3 steps at batch 8 × K=16, peak 132/188 G RAM,
      13.6 min/step. At batch 16 (256 traj) step 1 dies of CPU OOM; §6.
- [ ] **GRPO run in progress** — started 2026-08-27 04:22, 267 steps ≈ 60 h.
      Watch `val-core/…/acc` against train reward (divergence = reward
      hacking) and `evidence_iou`, which has the most headroom (0.141 of 0.5).
- [ ] Alias LLM enrichment + spot check
- [ ] base arm has no per-row dump; re-run needs the GPU back

## 9. Runbook

Every new terminal: `conda activate verl && cd ~/agentic-tvg` (LD_PRELOAD
rides along automatically; the training script's preflight guard backstops it).

**Step 1 — all data, one command** (~31G download + render, 30–60 min)

```bash
bash prepare_data.sh
```

CHECKs printed as it runs: every download `[ok]` · allocation
`joined 1157 | SFT: 240 forced + 360 sampled = 600 questions -> 1379 traces |
+ geminicot 600 | RL: 1,068` · rendered 1,958 rows →
`data/processed/{sft_train,sft_val}.parquet` + `frames/`.
Re-running is safe: downloads resume, populated dirs are skipped, parquet is
overwritten.

**Step 2 — smoke: 2 real training steps** (~7 min)

```bash
SMOKE=1 bash run_sft.sh
```

CHECK: finite, decreasing loss; no truncation errors; then
`rm -rf results/smoke` (~19G). (verl always validates and saves on the final
step regardless of the -1 freqs — expected.)

**Step 3 — the real run** (~6–8 h; `nvidia-smi` must read 0 MiB first)

```bash
tmux new -s sft        # detach: Ctrl-b d · reattach: tmux attach -t sft
bash run_sft.sh trainer.test_freq=25 optim.lr_warmup_steps_ratio=0.1
```

The script tees to `logs/sft_mix_<timestamp>.log` on its own — no redirection
needed. Ctrl-C inside the attached session kills the run; leave with Ctrl-b d.

test_freq=25 gives five val points across the ~124 steps — the overfitting
curve that decides whether EPOCHS moves. Watch `val/loss` at steps 25/50/75/100:
flat or rising between epochs → do not raise EPOCHS.

Everything the run produces lands in `results/<EXP_NAME with hyphens>/` —
`ckpt/` (the trainer writes checkpoints straight there), plus `curves.png`,
`metrics.csv`, `console.log` and the hydra config snapshot, all regenerated by
run_sft.sh's EXIT trap so a crashed run still leaves a readable curve. On a
resumed run the trap concatenates every attempt's log oldest-first, so the
curve is continuous across the break.

Metrics are persisted two ways (both automatic):
- tensorboard events → `logs/tb/<EXP_NAME>/` (run_sft.sh exports
  TENSORBOARD_DIR; `tensorboard --logdir logs/tb` if you want the live UI —
  VS Code forwards the port automatically)
- the console log itself → `python plot_sft.py <log> -o PNG --csv CSV`
  (loss / memory / lr / grad-norm / mfu panels; with no args it picks the
  newest log and writes the PNG beside it)

**Step 4 — GRPO** (~58 h at 785 s/step for 267 steps)

```bash
tmux new -s grpo
bash run_grpo.sh          # no arguments; MODEL_PATH defaults to the merged SFT model
```

Two things need a human during the run:

- **Delete the superseded checkpoint after every save** (every 20 steps ≈ 4.6 h).
  `ls results/grpo-vanilla/ckpt/` must show exactly one `global_step_*`; keep
  the one named in `latest_checkpointed_iteration.txt`. Each is ~17 GB and a
  save writes the new one before removing the old, so two of them plus the next
  save will not fit. `max_actor_ckpt_to_keep=1` handles this from the next run
  onward — the run started 2026-08-27 09:51 predates that fix and needs the
  manual pass.
- **Watch `val-core/…/acc` against train reward.** Training reward rising while
  val accuracy is flat is reward hacking; `response_length/mean` climbing off
  its 3.3K baseline is the usual mechanism. Plot any time with
  `python plot_grpo.py`.

To resume after an interruption: `bash run_grpo.sh` with no arguments — adding
`TOTAL_STEPS` would reshape the cosine schedule and make the lr jump. Then
delete the checkpoint it resumed from, once the next one lands.
