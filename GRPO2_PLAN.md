# GRPO Round 2 Plan — `grpo_v2` (2026-09-01; rewritten same day after the group-level analysis, all §3 changes implemented)

**OUTCOME (2026-09-05, full results in `V2_RESULTS.md`):** ran as designed.
Primary criterion: closing 3-mean acc 0.6214 vs baseline 0.6023 — **positive
(+1.9 pt) but short of the 1-SE clean-win bar**. The observable-only lines
overdelivered: evidence_iou 0.254 (terminal 0.269) broke v1's 0.21 plateau
and transferred externally (Charades R@0.5 +15.1 vs SFT). Curriculum and
saturation behaved as §3e predicted; RFT stayed neutral (×5). This file is
now a historical design record — read V2_RESULTS.md for what happened.

Design for the second GRPO run: **same start model as round 1
(`results/sft-mix/merged`), judge v2 as the one real reward change, an
epoch-boundary curriculum against pool saturation, constant lr as hygiene.**
Evidence base: `GRPO_v1_RESULTS.md` (round-1 forensics; its §4 group-level
saturation analysis, added 2026-09-01, is what reshaped this plan) and
`GRPO_NOTES.md` (mechanics). Two earlier routes were measured and dropped —
their post-mortems live where the evidence is:

- **rft2 / SFT2 with reflection-data injection** (this file's first draft,
  then V2_PLAN's variant): abandoned. `V2_PLAN.md` holds the post-mortem —
  the injected traces presuppose localisation the 128-frame budget cannot
  supply, and the RL pool's ≤302s videos make multi-crop unprofitable anyway.
- **"the plateau was the cosine lr"** (this file's first §3c): withdrawn.
  Learning speed vs lr is non-monotonic across round 1's phases and the
  plateau tracks pool saturation instead (GRPO_v1_RESULTS §4).

## 1. What round 1 established

Round 1 (`grpo-vanilla`, 267 steps) worked but plateaued: val acc
0.456 → 0.50–0.55 band from step 180, evidence_iou stuck at ~0.21, policy
converged to exactly one crop per trajectory (cap allows 3).

- **The plateau is pool saturation.** Groups the policy has mastered
  (mean acc ≥ 0.9 over the 16 rollouts) grow 10.9% → 29.5% across the run;
  groups with zero acc variance 10.3% → 21.4%; within-group reward spread
  −20%. The all-wrong share never moves (~4–5%) — easy prompts get used up,
  hard ones stay. Full trend + method: GRPO_v1_RESULTS §4 ("The pool
  saturates"), reproduced by `data_prep/analyze_groups.py --signal`.
  DATA.md §3 predicted exactly this failure mode for the 1,068-prompt pool.
- **The judge leaked half-credit.** The 2026-09-01 full-set audit (opus
  re-grading all 114 val rows) showed haiku's 0.5 was a *refuge verdict* in
  both directions: of its disputed PARTIALs, 11 deserved FULL and 5 deserved
  0, and 23.2% of all 34K training verdicts sat in the 0.5 bucket —
  flattened and partly *noisy* advantage exactly in the mid-difficulty
  groups where the within-group comparison carries the most signal. Fixed by
  judge v2 (§3a).
- **Wrong answers co-occur with missing the window.** val-rft error taxonomy
  (`results/val-rft/analysis.md`): of GRPO's wrong answers 25 were
  un-grounded vs 5 grounded; 18 answers were right *without* grounding.
  Localisation is where the headroom is — but it is a *capability* gap, not
  an incentive gap: within an acc-tied group iou already fully determines
  the ranking at any positive weight (§3b), and the policy still plateaued
  at 0.21.
- **One crop bounds iou** (31/114 val prompts iou ≡ 0 at all 13
  checkpoints), and multi-crop is out of scope for this round (§3d).

## 2. What round 1 ruled out: lr and batch as stability knobs

The train-reward sawtooth (step-to-step std 0.11–0.18) is **sampling noise
from 8 prompts/step, not lr instability**. Two independent proofs
(2026-09-01, from `metrics.csv` + `rollouts_grpo267/`):

1. lr decayed 8× across the run (1e-5 → 1.2e-6); oscillation amplitude did
   not shrink (0.113 → 0.176 — it grew slightly, tracking the widening
   between-prompt difficulty spread 0.351 → 0.388).
2. Zero-free-parameter prediction: between-prompt reward std 0.369,
   within-group std 0.344 → predicted per-step std
   √(0.369²/8 + 0.344²/128) = **0.134** vs observed 0.113–0.176.

Consequences: don't chase the sawtooth with lr or batch changes; read the
train curve as a 20-step moving average. Optimization health is judged by
grad_norm (smooth 0.05–0.075), entropy (smooth decline, no collapse), and
the val curve — all clean in round 1. Keep `TRAIN_BS=8`, `GROUP_SIZE=16`
(K is the within-group baseline reliability and must not shrink;
GRPO_NOTES §4; RAM ceiling rules out larger batch anyway).

Scope note: this section rules lr out as the *oscillation* driver. It says
nothing about the plateau — that attribution question is settled separately
by the phase-slope + saturation measurements in GRPO_v1_RESULTS §4 (also not
lr). "Raise the lr" is contraindicated by the same table: round 1's
highest-lr phase was its slowest-learning phase. A 30-step probe recipe for
a higher-lr cosine exists in the session notes if that is ever revisited;
it buys speed-to-plateau at best, not a higher plateau.

## 3. The changes (all implemented 2026-09-01; defaults in the scripts)

This is a **recipe-level comparison, not a single-axis ablation**: judge v2
+ constant lr + the epoch-boundary curriculum all change together, so
`grpo_v2` vs `grpo-vanilla` reads as "recipe v2 vs v1". Accepted knowingly —
the three changes point the same way and none is worth 60h of isolation.
The start model is shared with round 1, so round 1's +9 pt is inherited,
not re-earned.

Effect sizes, measured by re-scoring round 1's own 34,048 trajectories
(GRPO_v1_RESULTS §4 "Pre-flight"): judge v2 reorders within-group advantage at
Spearman 0.874 / 22.8% of groups change their best trajectory; the iou
re-weight 0.992 / 3.1%. **Round 2 is a judge round**; the curriculum's job
is to keep the signal alive long enough for that to matter.

### 3a. Judge v2 — DONE, and now the code default

Shipped in `agentic_tvg/judge_v2.py`: `claude-sonnet-5` (haiku+reasoning
topped out at 75% consistency), question-anchored rubric (unasked detail /
wording never demote; scene-description non-answers = 0), reasoning then a
`VERDICT:` line, cache `data/processed/judge_cache_v2.jsonl` (separate from
v1's; cache rows record model+rubric and mismatched rows never load).
Effect: PARTIAL 47 → 11 rows on the 114-row val.

**Selection is the `JUDGE_V` env var, default `2` since 2026-09-01**
(reward.py), exported and echoed by run_grpo.sh's `[recipe]` line so the
instrument is recorded in every console log. `JUDGE_V=1` reproduces v1
numbers. v1/v2 verdicts are never comparable; quote every acc with its
instrument version.

**v2 baselines already measured** (114-row rl_val, v1 in parens):
SFT 0.5395 (0.4561) · **GRPO 0.5965 (0.5044)** · RFT 0.5702 (0.5044).
Extended 2026-09-03: round-1's closing THREE checkpoints (240/260/267)
re-judged on v2 and re-scored under the qa2 reward —
`data_prep/rescore_rollouts_v2.py`, artifact
`results/grpo-vanilla/v2_rescore.json`: acc_v2 0.6228 / 0.5921 / 0.5921,
**3-checkpoint mean 0.6023**, reward_v2 mean 1.319. (267's fresh re-judge
0.5921 vs the documented single-point 0.5965 = −0.4 pt sampling drift —
the reason the endgame compares 3-point means.)

What v2 buys, precisely: **ranking correctness, not gradient magnitude.**
The 0.5 refuge was noise in both directions, so mid-difficulty groups were
partly ranked by judge noise; v2 replaces that with a decision. It does not
relieve saturation — simulated on round 1's rollouts it *raises* the
zero-acc-variance share (21.4% → 26.9% in the last band) because 15×FULL +
1×PARTIAL groups collapse to 16×FULL. The curriculum (§3e) is what
addresses saturation.

Selection experiment, terminal validation (opus-v2 agrees with shipped
sonnet-v2 on 103/114), and caveats — unchanged from the first draft:
opus-v2 validation covered only the RFT arm; rubric v2 was iterated on the
same 114 rows it was validated on (spot-check training-pool verdicts before
the paper); sonnet-v2 sits ~2 pts strict of an opus reference. The three
judge.py hardening fixes (retry on truncation, docstring, cache instrument
filter) are done. **Cost note: the machine wipe (2026-09-01) lost the v2
cache (295 rows) — the run starts cold: ~34K fresh sonnet-5 verdicts.
Estimate the API bill before launch.**

One more fix landed mid-stage-1 (2026-09-01 late; effective from the next
process start — stage 2 or any resume): the cache now refreshes
incrementally from the file before every miss and re-checks after every API
return, with first-verdict-wins enforced at load. Under the async agent
loop each worker previously saw only its own verdicts plus the file as of
its first load — the cold-cache step-0 val duplicated 47% of its judge
calls (146/309). Residual duplication window is one in-flight API call
(round-1's ~0.1% regime). judge.py (v1) is deliberately untouched — frozen
instrument. Regression tests: `tests/test_judge_cache.py`.

### 3b. Reward `compute_score_qa2` — DONE, smaller than first drafted

`R = 0.5·format + judge_acc + TIME_WEIGHT_V2 · evidence_iou`, with
`TIME_WEIGHT_V2 = 1.0` (round 1: 0.5). Range [0, 2.5] — the score scale is
not comparable across rounds; everything else about the dict (keys, val
metrics) is unchanged. `compute_score_qa` is frozen for round-1
reproduction (with `JUDGE_V=1`).

Two things the first draft asked for are deliberately absent:

- **The format flip** (`+0.5 if ok` → `0/−0.5`): dropped (user call,
  2026-09-01). It is a uniform constant shift; GRPO's group-normalized
  advantage is exactly invariant to it. The bonus form stays and the reward
  floor stays 0.
- **The multi-crop shaping term**: dropped with its enabler (§3d). Round 1
  sampled 22 multi-crop trajectories in 34,048, so it would be ≡ 0.

On the weight itself, measured (GRPO_v1_RESULTS §4): near-inert — within-group
advantage Spearman 0.992, 3.1% of group winners change, and **inside an
acc-tied group any positive weight yields the identical ranking**
(scale-invariance; 0/1,860 tie-break flips across 0.5…5.0). What 1.0 buys
is cross-tier authority: a 0.5 iou gap can overturn one acc tier (at 0.5 it
cannot). Kept at 1.0 with that understanding; it is not a grounding lever
and no iou target hangs on it (§5).

### 3c. lr schedule: constant — hygiene, not a lever

Constant 1e-5 (verl's default; run_grpo.sh sets it since 2026-09-01).
Why it is safe: round 1 ran ~50 steps at 1e-5 with its lowest grad_norm
(0.051) and gentlest entropy slope; entropy ended at 0.84 with the fastest
observed decline only −0.147/100 steps — far from collapse. Why it is
*only* hygiene: the plateau is saturation, not step size (§1), so no
unfreezing is expected from this. Its real benefits: no TOTAL_STEPS-as-
anneal-denominator, which matters twice now — the horizon is `EPOCHS` and
the stage-2 pool is smaller, either of which would bend a cosine curve at
the resume seam.

Collapse fallback unchanged (GRPO_NOTES §4 signature — entropy sustained
fall + grad_norm sustained rise): resume the latest checkpoint with
`actor_rollout_ref.actor.optim.lr_scheduler_type=cosine
actor_rollout_ref.actor.optim.min_lr_ratio=0.3`.

### 3d. Multi-crop: out of scope for round 2

The data-injection enabler (mixing `sft_longvideoreflection_3k` 2-crop
traces into SFT/RFT) was abandoned 2026-09-01 after measurement —
`V2_PLAN.md` has the full post-mortem. The short version: those traces
pinpoint a median 7 s window on ~944 s videos (a 124× narrowing; only
9/1,562 start coarse) — a move our 128-frame global view (7.4 s/frame
there) gives the model no evidence for — while the RL pool tops out at
302 s, where the first crop is usually right and round 1 rationally
extinguished second crops (22 sampled, mean score below group mean). The
prompt-nudge fallback goes with it (it would change the train+val prompt
distribution for a behaviour with no payoff in-pool).

`num_tool_calls` stays on the monitoring list as an observable, not a goal.
`extract_rft.py`'s exactly-one-crop parse stays as-is deliberately — with
multi-crop out of scope, re-freezing single-crop in a future RFT pass is
consistent, and the parse is one line to relax if scope changes.

### 3e. Epoch-boundary curriculum — ADOPTED (was: deferred to round 3)

The direct counter to §1's saturation. Run epoch 1 on the full 1,068-prompt
pool, then drop the prompts the policy has mastered and run epoch 2 on the
rest. The first draft deferred this as "changes the data axis; also epoch
math"; the horizon is per-epoch now, so the epoch-math objection is void,
and the data-axis objection is subsumed by the recipe-level framing (§3).

Calibration on round 1 (each prompt visited once per epoch; 1,044 prompts
with both visits): a single 16-rollout visit predicts next-epoch saturation
well —

| epoch-1 visit acc | n | mastered again in epoch 2 | zero-variance in epoch 2 |
|---|---:|---:|---:|
| ≥ 0.875 | 163 | 85.3% | 51.5% |
| 0.75–0.875 | 113 | 56.6% | 21.2% |
| 0.5–0.75 | 245 | 15.1% | 5.3% |
| < 0.5 | 523 | ~0% | — |

**Threshold: visit acc ≥ 0.9** (user call 2026-09-03, revising the drafted
0.75): the mid-band 0.75–0.9 still carries within-group variance (see
`results/grpo-v2/per_question_analysis.png`, middle panel), so only the
≥0.9 spike — 16/16-agreement territory — is dropped. On stage 1 that cut
217 of 1,064 questions; kept 851 (+4 unvisited) → 106 stage-2 steps. The
hard end is kept — all-wrong groups still carry iou/partial-credit
gradient. The cut is computed from **stage 1's own rollouts** (v2-judged),
never from v1 numbers (the calibration table above only shows the
predictor works).

Mechanics (all verified against verl 0.9.0 source):

1. Stage 1 `EPOCHS=1` ends at step 133 and **saves there** (`is_last_step`
   forces a save regardless of save_freq).
2. `python data_prep/filter_mastered.py --rollouts results/grpo-v2/rollouts`
   → `rl_train_ep2.parquet` (+ selection json listing every dropped
   question with its visit acc — skim it before stage 2). It prints the
   exact stage-2 launch line.
3. **`mv results/grpo-v2/ckpt/global_step_133/data.pt{,.bak}` before the
   resume.** The dataloader-state restore decides "at epoch boundary" using
   the NEW loader's length (133 % ~98 ≠ 0), so without this it loads a
   133-batch state into a ~98-batch loader. Removing data.pt hits the clean
   start-from-scratch branch.
4. Stage 2: **`bash run_grpo_stage2.sh`** — the wrapper derives everything
   from disk state (rows in the ep2 parquet → steps/epoch; registered
   checkpoint step → EPOCHS = step//steps_per_epoch + 1 and TOTAL_STEPS =
   step + steps_per_epoch) and does the data.pt surgery itself.
   `DRY=1 bash run_grpo_stage2.sh` prints the derived launch without
   running. Why EPOCHS must NOT be 1: verl computes `current_epoch =
   global_steps // len(new_dataloader)` on resume and loops
   `range(current_epoch, total_epochs)`; with the smaller pool the
   quotient is ≥1, so EPOCHS=1 makes the loop empty — the run exits
   cleanly after val_before_train with zero training steps (measured
   2026-09-03, cost one launch).

What carries across the seam: adapter + optimizer state (resume), and the
KL anchor — `ref_in_actor` references the base minus adapter, so both
stages stay anchored to the SFT policy. Val (full 114 rows every 20 steps)
is untouched and comparable throughout.

Still deferred: judge model up-tier (only if a training-pool calibration
audit fails), larger batch for curve smoothness (cosmetic; use the moving
average).

## 4. Order of operations

0. **Environment rebuild** (the box was wiped and re-cloned 2026-09-01;
   87G free): `.env` restored (ANTHROPIC_API_KEY for the judge, HF_TOKEN);
   `results/sft-mix/merged` pulled from the Hub (hf_pull.sh — in progress);
   `bash prepare_data.sh` for annotations + selfqa/rl_val videos + base
   model, then `render_traces.py`/`extract_rl.py` regenerate
   `data/processed/` (RL training decodes the global view from the selfqa
   mp4s at rollout time — the videos are required, not just the parquets).
   Geminicot only matters if SFT is ever re-run; skippable now.
1. **GPU lock** per protocol (`/tmp/gpu-owner.lock`; queue via SendMessage,
   never `ray stop --force`).
2. **Smoke** (~30 min, throwaway): `TOTAL_STEPS=3 EXP_NAME=smoke bash
   run_grpo.sh`, then: `[recipe]` line says `REWARD_FN=compute_score_qa2
   JUDGE_V=2`; judge v2 verdicts land in `judge_cache_v2.jsonl`; reward ≤
   2.5; and **read the judge $/step off the console** — 3 steps × 128
   trajectories is enough to extrapolate the ~34K-verdict bill before
   committing. The step-0 val it runs (114 rows) caches verdicts the real
   run will reuse. `rm -rf results/smoke`.
3. **Stage 1**: `EPOCHS=1 bash run_grpo.sh` (defaults: grpo_v2, qa2,
   JUDGE_V=2, constant lr, from results/sft-mix/merged) — 133 steps, ~30h.
   No pre-run gate (user call 2026-09-01): monitoring is in-flight, and
   checkpoints every 20 steps bound any loss. **First-hours reads, nothing
   stops for them**:
   - the step-0 val prints before training starts (`val_before_train`):
     acc should land ≈ **0.5395** (SFT model on the v2 instrument). A big
     miss is judge wiring, not the model — that one IS worth killing the
     run for, and it shows within the first ~20 min.
   - by the step-20/40 vals, sanity against round 1's opening: entropy
     slope ≈ −0.10/100 with grad_norm ~0.05 (much steeper + climbing
     grad_norm = §3c collapse signature → stop, fall back), format
     0.487–0.5, response_length ~3.2K flat. Train score is on the
     [0, 2.5]/v2 scale — compare slopes, not levels.
   (If raising the lr is ever revisited, a `TOTAL_STEPS=30` capped run at
   `lr=2e-5 cosine min_lr_ratio=0.3` is the probe recipe — §2.)
4. **Between stages** (§3e): filter_mastered → skim the dropped list →
   `mv .../data.pt{,.bak}` → launch stage 2 with the printed line (~22h).
5. **Merge + eval**: `python merge_adapter.py --ckpt results/grpo-v2/ckpt
   --out results/grpo-v2/merged --base results/sft-mix/merged`; delete ckpt
   after verifying. Checkpoint reads via `score_rollouts.py` /
   `analyze_rollouts.py` (val-rft analysis is the worked example).
6. **Final numbers**: VideoSIAH-Eval via `run_benchmark.sh` (~109G streamed
   in chunks, ~6–11h/model) for grpo_v2's final checkpoint AND
   grpo-vanilla (never run yet) — same judge version on both. Report it as
   an out-of-budget transfer benchmark: its videos average 1,688 s against
   our ≤302 s training regime and 128-frame view (V2_PLAN post-mortem, §2).
7. **Stage-3 decision** (rft_v2): only if grpo_v2's rollouts look worth
   distilling; `extract_rft.py` defaults already point at
   `results/grpo-v2/rollouts` with the `rft_v2` prefix, `run_rft.sh` at
   `results/grpo-v2/merged`.

All RAM/plasma/glibc settings unchanged — the ladder is solved, don't
touch it. Checkpoint transient is 2×17G; with 87G free the round-1 ENOSPC
mitigations are unnecessary.

## 5. Success criteria and monitoring

Targets (val, v2-judge scale, vs the §3a baselines; n=114 → SE ±4.7 pts,
so compare 3-checkpoint means, never single readings):

- **Primary: grpo_v2's closing 3-checkpoint mean acc_v2 above
  grpo-vanilla's measured 3-checkpoint mean 0.6023** (rescored 2026-09-03,
  `results/grpo-vanilla/v2_rescore.json`) **by a clear margin (~1 SE,
  ±4.7 single-point)** — i.e. ≥ ~0.65 for a clean win; matching ~0.60 is a
  wash. reward_v2 comparison anchor: 1.319. Expected mechanism: cleaner
  ranking in mid-difficulty groups (judge) + signal kept alive in stage 2
  (curriculum).
- **Test-during ladder**: val auto-fires every 20 steps (full 114 rows);
  the step-0 val and the step-20/40 reads (§4.3) are the first scheduled
  looks; then at the stage boundary and mid-stage-2, run
  `analyze_groups.py --signal` on the live `rollouts/` — stage 2's
  mastered share should sit well below round 1's late-run 29.5% (the §3e
  counterfactual says ~5% at the boundary). If it climbs back fast, the
  pool is smaller than the policy's learning rate — early-stop on the val
  plateau rather than re-cutting mid-stage.
- **Observables, no targets**: evidence_iou (capability-bound at ~0.21;
  any rise is upside), num_tool_calls (≡1.0 expected — §3d).
- **Mechanism check** (`analyze_rollouts.py` taxonomy vs grpo-vanilla's
  val dump): acc gains should come with the "wrong & un-grounded" cell
  shrinking or stable — shrinking into "grounded & correct" would be
  grounding improving as a side effect; growing would mean priors-only
  gains. Either is reportable; know which one happened.

Guard rails while running:

- entropy sustained fall + grad_norm sustained rise = collapse signature →
  §3c fallback (checkpoints every 20 steps bound the loss).
- `response_length/mean` leaving its ~3.2K flatline = hacking / plasma
  early-warning (GRPO_NOTES §3b).
- format_score mean should stay ≈ 0.487–0.5 as in round 1 (bonus form
  unchanged).
- train reward: read the 20-step EMA only (§2); note the [0, 2.5] scale —
  round-1 dashboards don't transfer.
- the `[recipe]` console line is the instrument record: EXP_NAME,
  REWARD_FN, JUDGE_V, MODEL_PATH, TRAIN_FILE, EPOCHS per launch — check
  it at every (re)start, especially that stage 2 shows the ep2 parquet.
