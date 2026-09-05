# V2 Results — the `grpo_v2` recipe (runs 2026-09-03..05)

Companion to `GRPO2_PLAN.md` (the design) and `GRPO_v1_RESULTS.md` (v1
forensics). Every number here is on the **v2 instrument** (judge v2 =
sonnet + question-anchored rubric; reward `compute_score_qa2`, scale
[0, 2.5]); v1 checkpoints were re-judged onto this scale before comparison
(`data_prep/rescore_rollouts_v2.py`). Terminology: v1 = grpo-vanilla era,
v2 = this recipe.

## 1. Headline

**v2 beat v1's closing accuracy with 60% of the training steps, and broke
the localisation plateau v1 was stuck at for its entire run.** RFT
(self-distillation) remained neutral across three attempts — an informative
negative reconciled with the paper in §5.

| closing 3-checkpoint means | **v2 (200/220/239)** | v1 (240/260/267, rescored) | Δ |
|---|---|---|---|
| val acc | **0.6214** | 0.6023 | **+1.9 pt** (positive; short of the 1-SE clean-win bar ~0.65) |
| evidence_iou | **0.254** (terminal 0.269, still rising) | ~0.21 (flat all run) | **+4.4 pt** |
| reward (qa2 scale) | 1.375 | 1.319 | +0.056 |

Recipe deltas vs v1 (all measured, GRPO2_PLAN §3): judge v2 (the one real
reward change; within-group advantage Spearman 0.874 vs v1 judging),
TIME_WEIGHT 1.0 (near-inert, kept), constant lr (hygiene), and the
**epoch-boundary curriculum** (stage 1 = full 1,068-prompt pool for 133
steps; drop visit-acc ≥ 0.9 prompts → stage 2 = 851 prompts for 106 steps).

## 2. Validation trajectory (14 checkpoints)

| step | stage | val acc | evidence_iou |
|---:|---|---:|---:|
| 0 | 1 · full pool | 0.509 | 0.158 |
| 20 | 1 | 0.491 | 0.157 |
| 40 | 1 | 0.504 | 0.153 |
| 60 | 1 | 0.518 | 0.180 |
| 80 | 1 | 0.548 | 0.198 |
| 100 | 1 | 0.557 | 0.202 |
| 120 | 1 | 0.579 | 0.192 |
| 133 | 1 · stage end | 0.583 | 0.214 |
| 140 | 2 · pruned pool | 0.570 | 0.217 |
| 160 | 2 | **0.627** | 0.237 |
| 180 | 2 | 0.618 | 0.239 |
| 200 | 2 | 0.618 | 0.248 |
| 220 | 2 | **0.640** | 0.244 |
| 239 | 2 · final | 0.605 | **0.269** |

Flat first 40 steps (v1-like), judge dividend through mid-stage-1, and a
second take-off after the curriculum cut (140→160 = +5.7 pt, the largest
jump of the run). Paired mid-run comparison at equal steps (60/80/100,
both on v2 scale): v1 and v2 tie at 0.5409 — the recipe's edge materialised
after step 100, i.e. in the judge-dividend late-stage-1 and the curriculum
stage. Full v1 mid-run rescore: `results/grpo-vanilla/v2_rescore_midrun.json`.

## 3. Train side

- Score 20-step means 1.27 → 1.39 over stage 1 (late slope +0.27/100 steps
  = 2× v1's fastest); stage 2 jumped to 1.45–1.47.
- entropy 1.115 → 0.749 (no collapse; compression accelerating — a round-3
  watch item), grad_norm 0.054 → 0.107, response_length flat ~3.2K (zero
  hacking signature), ~756 s/step.
- **Saturation** (the v1 plateau mechanism, GRPO_v1_RESULTS §4): v2 reached
  ~31% mastered groups at step ~130 — 2× v1's speed — validating the cut
  timing; the pruned stage-2 pool re-saturated to 33.8% by 239, so the
  one-pass design stopped where the signal ran out.
  Figures: `results/grpo-v2/per_question_analysis.png` (stage 1),
  `per_question_analysis_stage2.png` (stage 2); reproduce with
  `plot_per_question.py`.

## 4. Per-question analyses

One row per (question, visit): mean acc over the 16 rollouts, within-group
reward variance (the advantage's raw material), component stats —
`results/grpo-v2/per_question_stage1.csv` (1,064 visits) and
`per_question_stage2.csv` (848). Key facts:

- The ∩ law: within-group variance peaks at mid-difficulty (median ~0.26–
  0.28 at acc 0.4–0.5) and dies at both ends — the visual case for the
  curriculum cut.
- Stage-1 → stage-2 paired (844 questions): mean acc **+16.4 pt**, 79%
  of questions improved; the mid-difficulty bands gained most (+21 pt).
- 260 questions were newly mastered during stage 2 (141 from the 0.75–0.9
  band, 94 from 0.5–0.75, 22 from 0.25–0.5, 2 from <0.25); 140 of them
  went 16/16.

## 5. RFT (stage 3): five consistent reads of neutral — and why LongVT's worked

### 5.1 Our results

| run | recipe | val acc | vs grpo_v2 (0.6214) |
|---|---|---|---|
| v1 rft | v1 traces, lr 1e-4 | 0.5702 (rescored) | −2.6 pt vs v1-GRPO 0.5965 |
| rft_v2 | v2 traces, `score>1.5`+acc=1, lr 1e-4 | 0.5833 | −3.8 pt |
| **rft_v2b** | paper dual criterion (acc=1 **and** iou≥0.3), ≤4/q, **lr 2e-5** | **0.6184** | **−0.3 pt** |
| rft_v2b_e1 | same, 1 epoch | 0.6140 | −0.7 pt |

Plus the external read (§6): no RFT variant beats grpo-v2 on Charades
either (R@0.5: 52.4 → 46.6 / 49.9). Five reads, one sign, all within noise
of zero. Two sub-findings:

- **The early negatives were an lr artifact, not distillation damage.**
  Paired diff of rft_v2 vs its base: 81/114 val rows unchanged, 15↑/18↓ —
  borderline-row perturbation. lr 1e-4 → 2e-5 plus the tighter data closed
  the gap to −0.3.
- **Epoch count is not the issue**: 1-epoch ≈ 2-epoch internally; the
  1-epoch run recovered about half the external R@0.5 dip.

The mechanism is visible in every loss curve: val/loss flat at 0.96–0.98
from the first checkpoint — the student already assigns near-maximal
likelihood to its own high-reward traces. **Self-distillation with zero
policy gap has nothing to teach.**

### 5.2 Why LongVT's RFT gained +6 where ours gains 0

LongVT's own Table 2 (7B, 512 frames) localises their RFT gain precisely:

| LongVT-7B stage | VideoSIAH-Eval | benchmark average |
|---|---:|---:|
| SFT | 34.8 | 44.1 |
| RL | 35.9 (**+1.1**) | 46.6 |
| RFT | **42.0 (+6.1)** | 47.7 (+1.1) |

The +6 lives almost entirely in one column; the other five benchmarks move
−2.0…+1.4. Four stacked reasons, the last one decisive:

1. **Their RL under-harvested in-domain.** Their RL trained on the same
   ≤~300 s clips we measured, but their headline eval is 1,688 s videos —
   the RL gain didn't transfer (+1.1). RFT distills the successful
   *behaviour* (tool-call format, scan→crop→verify procedure, answer
   style), and behaviour transfers across duration better than policy
   gradients do. RFT collected what RL left on the table. Our GRPO was
   evaluated in-domain and banked +11 pt itself — no leftovers.
2. **Their cold start left the niche empty.** LongVT stage 1 is 247K
   generic mixed CoT (only ~19K tool traces); "supervise on successful
   in-domain trajectories" happens for the first time at their stage 3, so
   its marginal value is high.
3. **Scale**: 15,353 traces and a 7B student vs our 2.9K and 4B.
4. **We spent the RFT dividend at stage 1 — by design.** The DATA.md §1
   role inversion: our SFT cold start *is* LongVT's stage-3 RFT data
   (their model's doubly-filtered successful rollouts). Our pipeline
   therefore ran an RFT-style consolidation before RL ever started, and
   DATA.md §1 priced this in on 2026-08-25 as a disclosed confound ("RL's
   marginal gain will read smaller than the paper's… our own stage-3
   self-distillation must come from our own rollouts"). By our stage 3,
   both the data niche (2) and the harvestable headroom (1) were spent.

**Net:** the two pipelines book comparable total gains (theirs SFT→final
+7.2 on VideoSIAH; ours +11 on rl_val) at different stages. "RFT works" vs
"RFT is neutral" is an accounting difference downstream of where the
successful-trajectory supervision is placed — not a reproduction failure.

### 5.3 Extraction notes

`data_prep/extract_rft.py` gained explicit `--min-acc` / `--min-iou` gates
(the v1-era `score>1.5` no longer implies acc=1 under the qa2 scale —
fixed 2026-09-04). rft_v2b set: 8,403 candidates over 835 questions →
2,872 train / 52 val (`data/processed/rft_v2b_*`).

## 6. External benchmark: Charades-STA zero-shot grounding probe

Design (`run_charades_probe.sh`, `data_prep/prepare_charades.py`): 399
sampled test queries / 328 videos (~30 s, fully inside the F=128 budget),
natural "when does X happen" questions, **no format change** — the metric
is IoU(the policy's crop window, GT span) via the training reward code;
judge disabled (no API cost). Cross-domain zero-shot: the models never saw
Charades or span-output training.

| model | mean IoU | R@0.3 | R@0.5 | R@0.7 |
|---|---:|---:|---:|---:|
| sft-mix | 0.379 | 56.9% | 37.3% | 17.3% |
| **grpo-v2** | **0.482** | **71.7%** | **52.4%** | **28.6%** |
| rft-v2b | 0.475 | 70.7% | 46.6% | 26.6% |
| rft-v2b-e1 | 0.475 | 72.4% | 49.9% | 25.3% |

SFT→GRPO: **+10.3 mIoU / +15.1 R@0.5 ≈ 6 SE at n=399** — the internal iou
climb is a transferable localisation gain, not val-set overfitting. RFT
preserves it (R@0.5 −5.8 ≈ 2.3 SE, consistent with the internal
neutral-to-tiny-loss). Dumps: `results/bench-charades-*/val_rollouts/`.

VideoSIAH-Eval was **deliberately skipped**: its 1,688 s average duration
is out of the F=128 budget's regime (13.2 s/frame; a crop must beat a
1/4.27 narrowing to out-resolve the global view) — the scope argument is
archived in `V2_PLAN.md` §2.

## 7. Incidents & ops (6, all recovered; ~6 h total loss)

OOM ×3 (host-RAM: save+val-window stacking ×1, long-session baseline creep
×2 — see GRPO_v1_RESULTS §4 additions), judge 529-overload hard-stop ×1
(fixed: 6 exponential retries), verl resume epoch-skip empty-loop ×1
(fixed: `run_grpo_stage2.sh` derives EPOCHS/TOTAL_STEPS from disk), credit
zero-crossing near-miss ×1 (landed inside an OOM outage window; enable
auto-reload). Judge bill ≈ 23–24K sonnet calls ≈ $50–70 incl. all rescores
and the pre-fix duplicate era (cross-process cache blindness, 47% → ~0
after the incremental-reload fix, `tests/test_judge_cache.py`).

## 8. Takeaways

1. **The judge is the reward.** Replacing a hedging grader (23% refuge
   verdicts) with a decisive one was worth ~7× any weight change, measured
   before launch by re-scoring old rollouts (`analyze_groups.py
   --reward-ab`). Pre-flight reward changes offline before buying GPU time.
2. **GRPO dies of saturation, not lr.** Track mastered-group share, not
   loss curves; prune mastered prompts at epoch boundaries. The ∩ law
   (variance peaks at mid-difficulty) is the whole curriculum argument in
   one plot.
3. **Localisation was trainable after all** — with a clean ranking signal
   it rose past a plateau previously diagnosed as a capability limit, and
   the gain transfers zero-shot to an external benchmark (+15 R@0.5).
4. **RFT is a placement decision.** Successful-trajectory supervision pays
   wherever it lands first (our stage 1, LongVT's stage 3); running it
   twice pays nothing and, at high lr, slightly perturbs.
5. **Ops:** host RAM is the binding constraint (all 3 OOMs), long sessions
   creep toward the kill line (resume resets the baseline), and every
   background judge/cache/resume subtlety that can silently waste money
   eventually did — instrument the recipe line, dedupe the cache, anchor
   resume horizons to disk state.

## 9. Artifact index

| artifact | path |
|---|---|
| **final v2 model** | `results/grpo-v2/merged` |
| RFT ablation models | `results/rft-v2/merged`, `results/rft-v2b/merged`, `results/rft-v2b-e1/merged` |
| v2 train/val dumps + tb + curves | `results/grpo-v2/` |
| per-question CSVs + figures | `results/grpo-v2/per_question_*.{csv,png}` |
| v1 rescored baselines | `results/grpo-vanilla/v2_rescore{,_midrun}.json` |
| RFT evals | `results/val-rft-v2/`, `results/val-rft-v2b/` |
| Charades probe dumps | `results/bench-charades-*/` |
| RFT ablation summary (machine-readable, recomputed from dumps) | `results/rft_ablation_summary.json` |
| judge verdict cache (audit trail) | `data/processed/judge_cache_v2.jsonl` |
| stage-2 curriculum pool + dropped list | `data/processed/rl_train_ep2.parquet` + `.selection.json` |
| rft_v2b dataset + review | `data/processed/rft_v2b_*` |
| reproduction scripts | `run_grpo.sh`, `run_grpo_stage2.sh`, `run_rft.sh`, `run_charades_probe.sh`, `data_prep/{filter_mastered,analyze_groups,rescore_rollouts_v2,prepare_charades}.py`, `plot_per_question.py` |
