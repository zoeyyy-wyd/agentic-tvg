# GRPO Results — `grpo-vanilla` (2026-08-30)

Analysis of the production GRPO run (`run_grpo.sh`, config rationale in
`GRPO_NOTES.md`, budgets in `FRAMES_SWEEP.md` §5). Written at step 243 of 267;
latest checkpoint `results/grpo-vanilla/ckpt/global_step_240`. ~24 steps
(~5 h) remain; the final validation will land at step 260.

Sources: `results/grpo-vanilla/curves.png`, `metrics.csv`, console logs.

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

**Statistical caveat.** With n=114 the standard error on acc is ~±4.7 pts.
The overall +9.2 pt trend is ~2 SE and corroborated by the reward curve, but
the final 0.500 → 0.548 jump over the last 40 steps is within noise. Confirm
on the full test set with the step-240/260 checkpoint before quoting 0.548.

## 3. Training-side signals

- **Reward**: `critic/rewards/mean` 1.088 (first 20 steps) → 1.163 (last 20).
  Tracks val reward — no divergence, no hacking signature.
- **Entropy**: 1.11 → 0.88, smooth decline, no collapse; exploration intact.
- **KL to ref**: `actor/kl_loss` rises monotonically to 0.0128. With
  `kl_coef=1e-3` its loss contribution is ~1e-5 — normal drift, ignore.
- **pg_clipfrac / ppo_kl ≡ 0** all run. Expected, not a bug: train==mini batch
  means one fully on-policy update per step, so ratio ≡ 1 (GRPO_NOTES §3).
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
  ~0.19–0.22** — one crop bounds localisation precision. If iou matters,
  raise its reward weight or shape rewards to make a second crop worth the
  tokens.

## 4. The step-140 gap in metrics.csv

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
is the known plasma/DataProto RAM ladder (GRPO_NOTES §6).

## 5. Performance

- Median **756 s/step** (~12.6 min, as budgeted); ~920k tokens/step,
  throughput ~1,200 tok/s, actor MFU 0.34.
- Validation every 20 steps costs 860–1,290 s each.
- ~50 h of pure training for 240 steps, excluding validation.
- Host RAM sawtooths 100–130 GB — the expected allocation rhythm, never near
  the 188 GB ceiling. GPU: 39.3 GB allocated / 42.1 GB reserved, stable.

## 6. Recommendations

1. **Evaluate step-240/260 on the full test set** before quoting the final
   number (§2 caveat).
2. **Re-budget the reward**: drop or down-weight the saturated format term;
   spend it on evidence_iou if localisation is a goal.
3. **Incentivise multi-crop** if agentic grounding is the point — the current
   reward makes one crop "good enough" and the behaviour never emerges.
