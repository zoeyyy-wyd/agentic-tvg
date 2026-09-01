# GRPO Round 2 Plan — `grpo2` (drafted 2026-09-01)

Design for the second GRPO run, starting from the RFT-distilled model.
Evidence base: `GRPO_RESULTS.md` (round-1 forensics), `GRPO_NOTES.md`
(mechanics), the 2026-09-01 oscillation analysis (§2 below), and the judge
audit (`judge_audit.py`, GRPO_RESULTS §8.4).

## 1. What round 1 established

Round 1 (`grpo-vanilla`, 267 steps) worked but plateaued: val acc
0.456 → 0.50–0.55 band from step 180, evidence_iou stuck at ~0.21, policy
converged to exactly one crop per trajectory (cap allows 3). The bottleneck
is the **reward signal**, not optimization:

- **format term saturated at its 0.5 max from step 0** (4 decode failures in
  34K trajectories). Within a group it is a constant, so under GRPO's
  group-normalized advantage it contributes zero gradient.
- **The judge leaked half-credit.** The 2026-09-01 full-set audit (opus
  re-grading all 114 val rows — superseding the earlier 21-flip audit's
  "errs only lenient" reading) showed haiku's 0.5 was a *refuge verdict*
  in both directions: of its disputed PARTIALs, 11 deserved FULL and 5
  deserved 0, and 30% of all 31K training verdicts sat in the 0.5 bucket.
  Either way the effect on GRPO is the same: flattened advantage
  differences exactly in the mid-difficulty groups where the within-group
  comparison carries the most signal. Fixed by judge v2 (§3a).
- **Wrong answers co-occur with missing the window.** The val-rft error
  taxonomy (`results/val-rft/analysis.md`, acc × evidence_iou≥0.3): of
  GRPO's wrong answers 25 were un-grounded vs 5 grounded, while 18 answers
  were right *without* grounding (priors + global view). Localisation is
  where the headroom is — direct support for the §3b iou re-weighting.
- **One crop bounds iou.** 31/114 val prompts had iou ≡ 0 at all 13
  checkpoints; a second crop was never worth it under weight 0.5.
- **Difficulty is bimodal**: late-run, 21% of groups are 0/16 fully correct
  (gradient rides on partial credit + iou only) and 23% are 14–16/16
  (near-zero signal).

## 2. What round 1 ruled out: lr and batch size

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

## 3. The changes

One axis changes: **the incentive structure** — the reward signal (judge +
weights + shaping) plus the §3d prompt nudge that makes the shaped
behaviour sampleable at all, with the lr schedule as a free rider. Data,
lengths, KL, arch, batch/K all stay, so improvements are attributable to
that axis (the nudge and the shaping are entangled by design — they only
work together).

### 3a. Tighten the judge — **DONE 2026-09-01 (separate judge session),
and it went further than this plan asked**

Shipped in `agentic_tvg/judge.py` as instrument v2: `claude-sonnet-5` (not
haiku+rubric-line — haiku+reasoning topped out at 75% consistency),
question-anchored rubric (unasked detail / wording never demote;
scene-description non-answers = 0), reasoning then a `VERDICT:` line,
default cache `data/processed/judge_cache_v2.jsonl` (old cache untouched —
the file split resolves the cache-key trap). Effect: PARTIAL bucket
47 → 11 rows on the 114-row val. Note: Claude 5 API rejects `temperature`;
the param was removed.

**v2 baselines already measured** (114-row rl_val, v1 in parens):
SFT 0.5395 (0.4561) · **GRPO 0.5965 (0.5044)** · RFT 0.5702 (0.5044).
Quote every acc with its instrument version; v1/v2 numbers never mix.

**The selection experiment** (archived `results/val-rft/judge_audits/`, all
on the 114-row val): a 2×2 of model × rubric plus a reasoning ablation.
haiku-one-word vs opus-v1: 82%; haiku+reasoning: 75% (gap is model, not
format); sonnet-v1: 84% (model alone not enough); under rubric v2 haiku
still hedged (27 PARTIALs) while sonnet dropped to 11. Terminal validation:
**opus itself re-run under rubric v2 agrees with the shipped sonnet-v2 on
103/114 (90%)** — the highest agreement in the chain; residual
disagreements are genuine boundary calls, opus slightly more generous
(8:3, −2.2 acc pts).

Known caveats and pending fixes (assessed 2026-09-01, this session):
- The opus-v2 terminal validation covered only the **RFT arm's** answers;
  GRPO/SFT arms untested against opus-v2 (likely transfers; single-point
  evidence). Rubric v2 was also iterated on the same 114 rows it was
  validated on — spot-check a sample of *training*-pool verdicts before
  the paper.
- sonnet-v2 sits at the strict end of the pair: externally quoted v2 acc
  likely underestimates ~2 pts vs an opus reference.
- Determinism rests on the append-only cache, not on sampling: the Claude 5
  API takes no temperature, and even v1's cache held 45 same-key verdict
  flips (worker races). The three follow-up fixes are **DONE 2026-09-01
  (this session)**: (1) unparseable/truncated verdicts now retry with a
  doubled token budget instead of stopping the run (unit-tested: retry,
  exhaustion, and cache-hit paths; live smoke passed); (2) docstring no
  longer claims "temp 0"; (3) cache loading now skips rows whose recorded
  model/rubric don't match the live instrument (all 295 existing v2 rows
  pass the filter).

### 3b. Re-budget the reward — new `compute_score_qa2` in `reward.py`

Keep `compute_score_qa` untouched (comparability; `REWARD_FN` env switch
already exists). New function:

| term | round 1 | round 2 | rationale |
|---|---|---|---|
| format | `+0.5 if ok` | `0 if ok else −0.5` | **gradient-neutral by construction** — see note below the table |
| acc (judge) | {0, 0.5, 1} | same, rubric v2 | §3a |
| evidence_iou | `+0.5 × iou` | `+1.0 × iou` (`TIME_WEIGHT=1.0`) | iou is the plateaued target; with the stricter judge removing free half-credit, iou also becomes the main gradient source in all-wrong groups |
| multi-crop shaping | — | `+0.25 × max(0, iou_best − iou_first)` | pays only for a *productive* second crop; ≡ 0 for the current one-crop policy, so it is pure upside toward the behaviour that never emerged |

**Note on the format flip:** it changes no gradient at all. The new form is
a uniform −0.5 shift on every trajectory (ok: +0.5→0, fail: 0→−0.5; the
ok/fail gap stays 0.5), and GRPO's group-normalized advantage
`(r − mean)/std` is exactly invariant to a constant shift. The change is
pure bookkeeping, kept for two small reasons: `critic/rewards/mean` becomes
a pure task score (acc + iou terms only — directly comparable to the acc
curve, and no longer inflated by a saturated constant), and the explicit
−0.5 keeps the decode-failure margin visible. The gradient changes in this
round are the judge rubric, TIME_WEIGHT, and the shaping term — nothing
else. Dropping the format term entirely would also be near-equivalent
(4 failures in 34K trajectories); either is fine, deleting it is not worth
a code path divergence from `_base_score`.

Reward range shifts from [0, 2.0] to [−0.5, 2.25]; all dashboards re-scale.

### 3c. lr schedule: constant, not cosine

Round 1's plateau (step 180+) coincided with lr decaying below ~3e-6 — the
schedule froze learning during the phase where the new reward terms would
need it most. §2 showed decay buys no stability here. Round 2: verl's
default `constant` at 1e-5 (pass
`actor_rollout_ref.actor.optim.lr_scheduler_type=constant` through
`run_grpo.sh`'s `"$@"` — the script already sets `=cosine` earlier on the
same command line, and Hydra takes the last occurrence; if the installed
Hydra instead errors on the duplicate key, edit the two lr-scheduler lines
in the script directly. Verify `actor/lr` is flat in the first logged
steps either way). Side benefit: TOTAL_STEPS stops
being a schedule denominator, so extending or stopping early no longer
bends the lr curve (the round-1 resume footgun disappears).

Fallback if entropy falls fast + grad_norm rises (the collapse signature,
GRPO_NOTES §4): resume with cosine `min_lr_ratio=0.3`.

### 3d. Multi-crop reachability — the shaping term needs an enabler
(evidence added 2026-09-01)

Counted crop calls per trace across every data stage:

| stage | multi-crop share |
|---|---|
| LongVT sources we use: `rft_selftrace_15k3` / `sft_geminicot_4k8` | 5/15,354 and 1/4,881 (~0.03%) |
| LongVT `sft_longvideoreflection_3k` (**unused** — videos 253.6 GiB) | **54.5%** (1,637/3,004 have 2 crops) |
| our `sft_train.parquet` (1,923 traces) | **0%** — every trace exactly 1 crop |
| our `rft_train.parquet` | 0% **by construction** — extract_rft.py's structural parse requires "exactly one crop_video call" (DATA.md §8) |
| round-1 GRPO rollouts (34,048, temp-sampled) | 22 (0.065%) — 20 in the early half, 2 late; mean(score − group mean) = **−0.038** |

Three facts compound: the prior has essentially zero multi-crop support
(the one LongVT file that teaches crop→look→re-crop is the one we skipped
for disk); the RFT distillation structurally re-hardened single-crop; and
in round 1 the rare sampled multi-crop attempt scored *below* its group
mean, so GRPO actively extinguished it (20 → 2 across the run). The round-2
policy starts from the RFT model, i.e. with even less multi-crop mass than
round 1 started with. **The §3b shaping term therefore fires ~never on its
own** — it is kept (harmless, and correct if the behaviour appears), but it
needs one of these enablers:

- **A. Data injection (DECIDED 2026-09-01 — the primary enabler):** mix a
  subset of `longvideoreflection_3k`'s 2-crop traces into the RFT stage.
  Feasibility confirmed the same day:
  - The 1,637 2-crop traces map to **330 unique videos** (1,562 traces
    with parseable paths; one malformed path accounts for the rest), all
    present across `longvideoreflection_1..27.zip` on
    `longvideotool/LongVT-Source`.
  - Every zip entry is **ZIP_STORED (uncompressed)**, so single mp4s can
    be pulled by byte-range through `HfFileSystem` — no need for the
    253.6 GiB archives. Video→(zip, size, trace-count) map saved to
    `data/processed/reflection_video_map.json`.
  - Budget curve (greedy by traces/GiB): **~5 GiB → 307 traces / 44
    videos; ~10 GiB → 456 / 70; ~15 GiB → 576 / 88**. Recommended tier:
    ~10 GiB (456 traces ≈ 17% of a combined rft2 mix). Disk fits (45G
    free; delete mp4s after rendering, as with geminicot).
  - **Re-render into OUR template** (system prompt, strict-format user
    instruction, 128-frame global view, tool schema) via
    `render_traces.py` — needs a small extension to render two crop
    windows per trace, and the same review gate as the RFT data. Because
    the rendered prompt dialect is ours, the prompt distribution does NOT
    change → round-1 comparability survives.
  - Quality caveat (sampled 2026-09-01): the reflection traces are
    synthetic wrong-window + right-window pairs — the second crop's
    timestamp appears without derivation (e.g. first crop 260–265s, second
    2648–2673s, narration never explains the jump). They teach the retry
    *format*, not a search strategy. That is exactly what unlocking
    needs; don't expect them to teach clever search.
- **B. Prompt nudge (fallback only):** extend the system prompt to invite
  re-cropping. Kept in reserve in case A's traces fail review — it changes
  the train AND val prompt distribution, which would force a new-prompt
  baseline and muddy round-1 comparisons.
- Also fix forward: future `extract_rft.py` passes must not hard-require
  single-crop, or every RFT cycle re-freezes the behaviour.

### 3e. Deferred (do NOT bundle into this run)

- **Difficulty rebalance** — dropping the ~23% of prompts that were ≥14/16
  correct late in round 1. Real gains available, but it changes the data
  axis and breaks attribution; also epoch math. Round 3 candidate, using
  the per-prompt stats already computed from `rollouts_grpo267/`.
- **Judge model up-tier** — only if §4.2 calibration fails.
- **Larger batch for curve smoothness** — cosmetic (§2); costs linear wall
  time; use the moving average instead.

## 4. Order of operations

1. **GPU lock** per protocol (`/tmp/gpu-owner.lock`; queue via SendMessage,
   never `ray stop --force`).
2. **Build the reflection subset** (no GPU needed): fetch the ~10 GiB /
   70-video / 456-trace tier by byte-range from the reflection zips using
   `data/processed/reflection_video_map.json`; extend `render_traces.py`
   to render two crop windows per trace; render into our template; delete
   the mp4s after rendering (geminicot precedent). **Review gate**: user
   reads a sample of rendered traces (rft_review precedent) before any
   training touches them.
3. **Train `rft2`** = `rft_train.parquet` (2,237) + reflection subset
   (~456, ≈17% of the mix), from `results/grpo-vanilla/merged`, via
   `run_rft.sh` with new `TRAIN_FILE`/`EXP_NAME`. ~2h at the measured
   ~70 s/step. Context: the plain single-crop RFT run has **already been
   merged and evaluated** (2026-09-01) and came out neutral-to-weak vs
   GRPO — v2 acc 0.5702 vs 0.5965, iou 0.2206 vs 0.2178, tool calls ≡ 1.0
   — so it is the no-reflection ablation point, and rft2's justification
   is the **behaviour unlock**, not an acc gain. Disk note: RFT saves
   crashed twice on ENOSPC (needs 2×19G transiently); with only 45G free,
   fetch/render/delete the reflection videos in batches.
4. **Merge + gate eval**: `merge_adapter.py --ckpt results/rft2/ckpt
   --base results/grpo-vanilla/merged --out results/rft2/merged`, then one
   `val_only` run (judge v2 is the live default; v1 comparability comes
   from the frozen v1 numbers). Two readouts decide the round-2 start:
   - `num_tool_calls/mean` — the unlock. If still ≡ 1.0 after SFT on 17%
     2-crop data, escalate to §3d option B (prompt nudge) before burning
     60h of RL.
   - v2 acc — if rft2 lands well below GRPO's 0.5965 (beyond ~1 SE), start
     round 2 from `results/grpo-vanilla/merged` instead and accept the
     weaker multi-crop prior (or raise the reflection share and retrain —
     it is a ~2h loop).
5. **Implement** `compute_score_qa2` (§3b: format→penalty,
   TIME_WEIGHT=1.0, multi-crop shaping; judge v2 comes along for free as
   the live default); unit-test against a handful of cached transcripts.
   The three §3a judge.py fixes are already done (2026-09-01).
6. **Launch**:
   ```bash
   REWARD_FN=compute_score_qa2 EXP_NAME=grpo2 \
   MODEL_PATH=results/rft2/merged \
   bash run_grpo.sh actor_rollout_ref.actor.optim.lr_scheduler_type=constant
   ```
7. **Final numbers** (after the run): per-checkpoint reads via
   `data_prep/score_rollouts.py` (means) and `data_prep/analyze_rollouts.py`
   (error taxonomy + paired flips vs a reference dump — the val-rft
   analysis is the worked example); the shipping benchmark is
   **VideoSIAH-Eval** (652 open-ended QA, answer-acc only — no GT windows,
   evidence_iou ≡ 0 there) via `run_benchmark.sh`, which streams the ~109G
   video set through the small disk in zip-sized chunks, ~6–11h per model,
   resume-safe. Budget one pass for grpo2's final checkpoint and, if not
   already done, one for grpo-vanilla as the comparison point — same judge
   version on both.
   `TRAIN_FILE` stays `rl_train.parquet` (the RFT parquet is SFT traces,
   not RL prompts). TOTAL_STEPS=267 (same 2-epoch horizon; with constant lr
   it is now only a stopping point — the overshoot asymmetry argument from
   run_grpo.sh still favours sizing long). All RAM/plasma/glibc settings
   unchanged — the ladder is solved, don't touch it.

## 5. Success criteria and monitoring

Targets (val, new-judge scale, vs the §4.4 re-judged baselines):

- **evidence_iou off the plateau: > 0.30** (round 1: 0.21) — the primary
  goal; both new weight and shaping aim here.
- **num_tool_calls/traj > 1.2** — the multi-crop behaviour actually
  emerging (round 1: 1.00 flat).
- **v2 acc above 0.5965** (grpo-vanilla's closing checkpoint on the v2
  instrument) **by > 1 SE (±4.7 pts)** — remember n=114: single readings
  are noise, compare 3-checkpoint means (GRPO_RESULTS §2 caveat).
- **Mechanism check** (`analyze_rollouts.py` taxonomy): if the iou work is
  doing what it claims, the "wrong & un-grounded" cell (25 rows for
  grpo-vanilla) should shrink into "grounded & correct" — acc gains with
  that cell unchanged would mean better priors, not better grounding.

Guard rails while running:

- entropy sustained fall + grad_norm sustained rise = collapse signature →
  §3c fallback.
- `response_length/mean` leaving its ~3.2K baseline = hacking / plasma
  early-warning (GRPO_NOTES §3b). A second crop adds ~4.6K legitimately —
  expect drift toward ~5–8K *if* multi-crop emerges; runaway to the 16,384
  ceiling is the pathological version.
- train reward: read the 20-step EMA only (§2).
- format penalty: should stay ≈ 0; a persistent negative mean means the
  penalty-only change broke something — revert to bonus form.
