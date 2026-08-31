# GRPO Results — `grpo-vanilla` (2026-08-30)

Analysis of the production GRPO run (`run_grpo.sh`, config rationale in
`GRPO_NOTES.md`, budgets in `FRAMES_SWEEP.md` §5). §§1–7 were written at step
243 of 267. The run finished 267/267 on 2026-08-30 — after one more host-RAM
OOM during step 267 and a resume from `global_step_260` — final checkpoint
`ckpt/global_step_267`, pushed to HF `zoeyyy-wyd/agentic-tvg-grpo-final`.
§8 (post-run) analyses the two closing validations and the final dip.

Sources: `results/grpo-vanilla/curves.png`, `metrics.csv`, console logs, and
the per-trajectory dumps `rollouts/*.jsonl` / `val_rollouts/*.jsonl`.

## 1. Headline

**Training works and is healthy.** Over 240 steps, val accuracy rose from
0.456 to **0.548 (+9.2 pts)** and val reward from 1.021 to **1.152**, with no
entropy collapse and no sign of reward hacking (train and val reward move
together). The gains are all in answer accuracy; **evidence_iou barely moved
(0.155 → ~0.21, plateaued)** and the policy settled into exactly one crop
call per trajectory — the multi-crop agentic behaviour did not emerge.

## 2. Validation trajectory

Val set: 114 prompts, `test_freq=20`. Half-credit scoring makes acc move in
1/228 quanta.

| step | reward | acc | format | evidence_iou | tool_calls/traj |
|---|---|---|---|---|---|
| 0 | 1.021 | 0.456 | 0.487 | 0.155 | 0.98 |
| 20 | 1.024 | 0.461 | 0.487 | 0.154 | 0.97 |
| 40 | 1.004 | 0.439 | 0.491 | 0.149 | 0.99 |
| 60 | 1.074 | 0.500 | 0.491 | 0.165 | 0.99 |
| 80 | 1.060 | 0.469 | 0.496 | 0.190 | 0.99 |
| 100 | 1.081 | 0.500 | 0.491 | 0.179 | 0.98 |
| 120 | 1.067 | 0.474 | 0.500 | 0.186 | 1.00 |
| 140 | 1.100 | 0.496 | 0.496 | 0.218 | 0.99 |
| 160 | 1.102 | 0.496 | 0.500 | 0.212 | 1.00 |
| 180 | 1.097 | 0.500 | 0.500 | 0.194 | 1.00 |
| 200 | 1.109 | 0.522 | 0.491 | 0.192 | 0.98 |
| 220 | 1.104 | 0.500 | 0.496 | 0.218 | 0.99 |
| 240 | **1.152** | **0.548** | 0.500 | 0.208 | 1.00 |
| 260 | 1.152 | 0.539 | 0.500 | 0.225 | 1.00 |
| 267 | 1.113 | 0.504 | 0.500 | 0.218 | 1.00 |

**Statistical caveat.** With n=114 the standard error on acc is ~±4.7 pts.
The overall +9.2 pt trend is ~2 SE and corroborated by the reward curve, but
the final 0.500 → 0.548 jump over the last 40 steps is within noise, and so
is the closing 0.548 → 0.504 dip — see §8 for the forensics. Confirm on the
full test set before quoting any single checkpoint's number.

## 3. Training-side signals

- **Reward**: `critic/rewards/mean` 1.088 (first 20 steps) → 1.163 (last 20).
  Tracks val reward — no divergence, no hacking signature.
- **Entropy**: 1.11 → 0.88, smooth decline, no collapse; exploration intact.
- **KL to ref**: `actor/kl_loss` rises monotonically to 0.0128. With
  `kl_coef=1e-3` its loss contribution is ~1e-5 — normal drift, ignore.
- **pg_clipfrac / ppo_kl ≡ 0** all run. Expected, not a bug: train==mini batch
  means one fully on-policy update per step, so ratio ≡ 1 (GRPO_NOTES §4).
- **LR**: cosine 1e-5 → 1.2e-6. **grad_norm**: mild creep 0.05 → 0.075.

### Three spikes, one cause (steps 89, 195, 197)

pg_loss spikes (0.65 / 0.49 / 0.47), advantage-mean dips to −0.6, and
`response_length/max` at 16,195–16,373 all coincide: individual rollouts ran
away to the `MAX_RESP_LEN=16384` ceiling, scored badly, and produced large
negative advantages. `aborted_ratio` stayed 0 and training recovered within a
step. Harmless at this frequency (3 in 243 steps).

### Saturated / flat signals

- **format_score is pinned at ~0.5 (its max) from step 0.** Only 4
  tool-decode failures in the entire run. This reward term is saturated —
  it adds a constant, contributes no gradient. Candidate for removal or
  down-weighting next run.
- **Exactly one crop call per trajectory** (train and val; `num_turns` ≡ 4:
  prompt → crop → tool return → answer). The cap allows 3 calls; the policy
  never uses a second. This is the likely reason **evidence_iou plateaued
  ~0.19–0.22** — one crop bounds localisation precision (per-trajectory
  breakdown in §4). If iou matters, raise its reward weight or shape rewards
  to make a second crop worth the tokens.

## 4. Per-prompt rollout accuracy

From the per-trajectory dumps: `rollouts/<step>.jsonl` (train, 8 prompts ×
K=16 per step), `val_rollouts/<step>.jsonl` (114 prompts × 1 sample per
checkpoint). `acc` is half-credit: 0 / 0.5 / 1.

### Train side: difficulty is bimodal

| trajectories | steps 1–20 | steps 224–243 |
|---|---|---|
| acc = 1 | 36.8% | 42.6% |
| acc = 0.5 | 23.1% | 23.9% |
| acc = 0 | 40.1% | 33.4% |
| mean | 0.484 | 0.546 |

Grouped by prompt (160 groups per 20-step window), the count of
fully-correct rollouts out of 16 is U-shaped: in the late window **34 groups
(21%) have zero fully-correct rollouts** while 37 (23%) have 14–16 of 16.
Mid-difficulty prompts — where GRPO's within-group comparison is most
informative — are the minority.

**Zero-variance (no-gradient) groups: 0/160.** Even all-wrong groups differ
in half-credit and evidence_iou, so every group carries gradient — but for
the all-wrong fifth of prompts that gradient comes entirely from iou and
partial credit, never from a correct answer.

### evidence_iou: what "plateaued" means per trajectory

- Late-window train rollouts: **31% of trajectories have iou = 0** (early:
  40%); only 18% exceed 0.5; mean 0.25.
- Val: **31/114 prompts have iou ≡ 0 at all 13 checkpoints** — on these the
  model never once cropped the right place; only 15 prompts settle above
  0.5.
- Accuracy rises while iou stays flat: answers frequently do not require
  precise localisation — the 128-frame global view plus priors suffice, and
  the single crop caps achievable precision.

### Val per-prompt trajectories (114 prompts × 13 checkpoints)

- **11 prompts always correct, 10 never correct, 14 always half-credit.**
- Step 0 → step 240: 71 wrong→wrong, 22 right→right, **18 wrong→right,
  3 right→wrong** — net +15, so the improvement is real, not sampling luck.
- Stability caveat: 24 prompts flip correct↔wrong ≥4 times across the 13
  evals (@1 sampling noise). Comparing first-3 vs last-3 checkpoint means,
  7 prompts are clearly better and 0 clearly worse.

The gains concentrate in mid-difficulty prompts. Two cohorts did not move:
the 10 never-correct prompts (worth cross-checking against the 31 iou≡0
prompts) and localisation itself — the direct motivation for
recommendations 2–3.

## 5. The step-140 gap in metrics.csv

Step 140's row has only the 12 val columns filled; all 107 training columns
are empty. This is the crash/resume seam, not a logging bug:

1. The 2026-08-28 run completed the step-140 update and saved
   `global_step_140`, then was killed by the **Ray host-RAM OOM killer**
   before printing the step-140 metrics line (verl logs one combined line per
   step, after validation). Last line logged: step 139. Those training
   metrics were never written anywhere — tb events included — and are gone.
2. The resume (`resume_mode=auto`) loaded `global_step_140`, ran
   `val_before_train`, and logged a **val-only step:140 line** — hence the 12
   filled columns — then continued from step 141. `val_rollouts/140.jsonl`
   is that initial validation's dump.

One lost training step out of 243; no impact on conclusions. The OOM itself
is the known plasma/DataProto RAM ladder (GRPO_NOTES §3).

## 6. Performance

- Median **756 s/step** (~12.6 min, as budgeted); ~920k tokens/step,
  throughput ~1,200 tok/s, actor MFU 0.34.
- Validation every 20 steps costs 860–1,290 s each.
- ~50 h of pure training for 240 steps, excluding validation.
- Host RAM sawtooths 100–130 GB — the expected allocation rhythm, never near
  the 188 GB ceiling. GPU: 39.3 GB allocated / 42.1 GB reserved, stable.

## 7. Recommendations

1. **Evaluate the final `global_step_267` checkpoint on the full test set**
   before quoting a number (§2 caveat). §8 shows 240/260/267 are
   statistically equivalent on val, so shipping the final (only surviving)
   checkpoint is fine.
2. **Re-budget the reward**: drop or down-weight the saturated format term;
   spend it on evidence_iou if localisation is a goal.
3. **Incentivise multi-crop** if agentic grounding is the point — the current
   reward makes one crop "good enough" and the behaviour never emerges.
4. **Tighten the judge rubric** next run: add "restating the question's
   premise is INCORRECT", or move `JUDGE_MODEL` up-tier (§8, item 4). Doing
   either shifts the acc scale down, so re-judge historical checkpoints
   before comparing across runs.

## 8. Post-run: the closing dip (steps 260 → 267)

Final val landed at reward 1.113 / acc 0.504, down from the step-240 peak
(1.152 / 0.548). Forensics on whether the dip is real, from the aligned
per-sample diff of `val_rollouts/260.jsonl` vs `267.jsonl` and a
second-opinion judge audit. Verdict up front: **plateau noise, not
degradation — and not the judge.**

1. **Trajectory context.** Since step 180 val acc has oscillated
   0.500–0.548 — one ±1 SE band (±4.7 pts). 267's 0.504 equals the
   step-180/220 readings; 240 was the top of the band, not a trend.
2. **Decomposition.** The −0.042 reward delta is acc −0.035, iou −0.007,
   format 0.000. Under greedy decoding 47/114 outputs changed textually
   (late-run weight motion is tiny — lr ~1e-6, KL 0.012 — but enough to flip
   borderline tokens). 21 samples flipped acc: 14 down, 7 up, 18/21
   involving the PARTIAL grade — borderline answers reshuffling, not hard
   failures. 11/21 flips changed only wording on an unchanged crop window;
   4 also degraded the window (worst: idx 101, iou 0.50→0, answer went from
   near-verbatim GT to "playing with a frisbee").
3. **Judge determinism holds.** All 228 verdicts were judge-graded
   (`judge_used=1`); the 55/114 samples whose answer text was identical at
   both steps received identical grades, 0 inconsistencies — temp-0 plus the
   append-only cache does what it promises. Every flip corresponds to a real
   text change.
4. **Second opinion (`judge_audit.py`, repo root).** Re-graded all 21
   flipped samples' 260 and 267 answers with claude-opus-5 (same LongVT
   rubric, reasoning allowed): 79% agreement with haiku, and the dip
   *grows* under the stronger judge (−4.5 acc pts vs haiku's −4.0; as val
   acc, −0.040 vs −0.035). On the 267 answers opus never graded above haiku
   (0 higher, 4 lower) — haiku was lenient toward 267, not harsh. Where
   haiku errs at all, it errs lenient on the PARTIAL boundary, awarding 0.5
   to answers that restate the question's premise (idx 19) or contradict
   the reference (idx 99); that bias is constant across checkpoints, so
   relative comparisons stand (rec 4).

Bottom line: `global_step_267` ≈ `global_step_240` within measurement
noise. Quote "val acc ~0.50–0.55, plateaued since step ~180", not a single
point.
