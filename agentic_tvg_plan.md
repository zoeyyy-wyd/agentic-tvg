# Agentic-TVG Project Plan

**Multi-turn tool-calling Temporal Video Grounding (TVG) with Qwen3-VL + verl**

Version: v1.0 | Date: 2026-08-18 | Status: not yet started

---

## 1. Positioning

### 1.1 One-line description

Study the role of agentic multi-turn tool calling plus verifiable rewards (RLVR) on pure temporal video grounding. The model runs a "coarse global scan -> local close-up -> correction" loop over the video via a `crop_video(start_time, end_time)` tool, and finally emits the time interval `[t_s, t_e]` of the target event.

### 1.2 Relation to LongVT

LongVT (arXiv:2511.20785) targets long-video open-ended QA; TVG is only one component of its reward, and all of its conclusions rest on Qwen2.5-VL. This project does two things LongVT did not:

1. **Task decomposition**: narrow the agentic paradigm down to pure TVG. TVG ground truth is a time interval, so the reward is fully verifiable by IoU with no LLM-as-a-Judge required. That is a standard RLVR setting and fits single-GPU training.
2. **Re-examining the conclusions**: LongVT's three findings on Qwen2.5-VL — (a) cold-start SFT is indispensable, (b) tool reward is unnecessary, (c) IoU reward beats Recall — have never been checked on Qwen3-VL, whose native tool-calling and temporal grounding abilities are substantially stronger. That is this project's core research question.

### 1.3 Design principles

- **Zero data construction**: all training data comes from LongVT's official release (HuggingFace: `longvideotool/LongVT-Parquet`); evaluation uses standard benchmarks. The entire engineering budget goes into training strategy and reward design.
- **Single-GPU feasible**: every configuration is designed against a 1x A100 80GB constraint (COSMOS srv4-lg2).
- **No cross-generation comparison**: no comparison against the Qwen2.5-VL family. Our own zero-shot baseline is the anchor, and results are reported as relative improvements.

---

## 2. Tech stack

| Component | Choice | Notes |
|---|---|---|
| Base model | **Qwen3-VL-4B-Instruct** | Not the Thinking variant (handling reasoning content in the chat template during multi-turn rollout is a documented verl pitfall); not 2B (grounding ability is insufficient) |
| RL framework | **latest official verl release** | Multi-turn tool calling via the SGLang rollout path (the first-class citizen of verl's multi-turn support); not the LongVT fork (hard-wired to Qwen2.5-VL) |
| Rollout engine | SGLang (latest) | Co-located on the same GPU as the policy |
| SFT framework | LLaMA-Factory (preferred) or verl's FSDP SFT trainer | LLaMA-Factory usually adapts to Qwen3-VL fastest |
| Dependency versions | transformers >= 4.57, pins matching verl main | Ground truth is whatever makes verl's requirements + its bundled multi-turn VLM example (geo3k) run |
| Environment | New conda env on srv4-lg2 (e.g. `agentic_tvg`) | Leave the existing `trl` / `dpo` envs untouched |

**Tool interface** (keeping LongVT's minimal design):

- A single tool `crop_video(start_time: float, end_time: float)` returning frames resampled from that interval
- Interaction format: `<think>` -> `<tool_call>` -> `<tool_response>` -> think again -> (optionally call again) -> `<answer>[t_s, t_e]</answer>`
- Max turns T = 3
- Tool calls follow Qwen3-VL's native hermes-style format

**Frame and token budget** (the single-GPU bottleneck; enforced from both ends: system prompt declaration + processor configuration):

- Coarse global scan: 32 frames, low resolution (low `min_pixels`)
- Each crop returns: 16 frames, higher resolution
- Total context <= 12K tokens

---

## 3. Step 0 — Zero-shot agentic probe (mandatory before training)

**Purpose**: Qwen3-VL's native ability is already strong, which turns cold-start SFT from a necessity into a variable whose dosage must be determined. Use pure inference experiments to fix that dosage, and produce the report's first table along the way.

**Setup**: serve Qwen3-VL-4B-Instruct with SGLang, attach the crop_video tool, and test three modes on Charades-STA plus a long-video subset:

| Mode | Description |
|---|---|
| (a) Direct | No tools, emit the time interval directly |
| (b) Tool-optional | Tool offered in the prompt, calling is not forced |
| (c) Tool-forced | At least one call is forced |

**Metrics observed**: mIoU, tool call rate, and post-call window correction rate (IoU change of the second proposal relative to the first).

**Decision rule**:

- Call rate > 50% and formatting broadly correct -> lightweight SFT (2K traces)
- Chaotic calling behaviour / broken formatting -> full SFT (6.4K traces)

---

## 4. Data plan (zero construction)

### 4.1 Sources

| Use | Data | Scale | Effort |
|---|---|---|---|
| SFT cold start | Qwen-distilled temporal grounding iMCoTT traces from LongVT-Parquet | 6,395 items | Re-rendering script (see 4.2) |
| RL training | (video, query, GT window) triples from the LongVT-Parquet RL split | ~1.6K, topped up to 3–5K from the grounding subset | Extraction script (a few dozen lines) |
| Evaluation | Charades-STA (direct download of the official AllenAI package) + a long-video held-out set split from LongVT data | — | Download and use |

**Explicitly excluded**: ActivityNet-Captions (videos must be scraped from YouTube, high failure rate); LongVT's 220K non-tool CoT items (they serve general QA and are irrelevant here).

### 4.2 The only real data work: re-rendering the traces

LongVT's SFT traces are stored under the Qwen2.5-VL chat template. A script is needed to extract the `<think>` / `<tool_call>` / `<tool_response>` content and re-render it under the Qwen3-VL template (hermes-style tool calls). Pure text transformation, roughly one day of work. **This is mandatory** — SFT with a mismatched template would damage the model's native tool-calling ability.

### 4.3 Storage

Video files: LongVT repo download script + the official Charades package. Reserve 150–200 GB on the server.

---

## 5. Training pipeline

### Stage 1 — SFT cold start (dosage decided by Step 0)

- LoRA r=16, dosage of either 2K or 6.4K traces
- Only three training objectives: propose a window, read the frames returned by a crop, and correct a wrong window
- Estimated half a day to two days (single GPU)

### Stage 2 — GRPO (the core stage)

**Reward function** (verl function-based reward, a Python function inside the sandbox):

```python
def compute_reward(pred_span, gt_span, format_ok):
    r_fmt = 0.5 if format_ok else 0.0
    iou = temporal_iou(pred_span, gt_span)
    return r_fmt + iou   # swapped for the penalty-aware version during ablation
```

**Single-GPU configuration**:

| Item | Value |
|---|---|
| Training mode | LoRA RL (policy + SGLang engine on the same GPU) |
| gpu_memory_utilization (rollout side) | ~0.45 |
| Group size | 8 |
| Train batch | 32–64 prompts |
| Max turns | 3 |
| Temperature | 1.0 |
| Steps | 100–200, stop once reward saturates |

**Difficulty filtering**: run K=8 rollouts offline before training and drop samples that are all-correct or all-wrong (zero-variance group handling, following GSM8K/GRPO practice). In TVG, a group whose rollouts all score IoU~0 is far more common than in math problems, so this step cannot be skipped.

### Stage 3 — RFT self-distillation (optional)

Filter RL rollout trajectories with IoU >= 0.5 and feed them back into SFT. Cut this if time runs short and record it as future work.

---

## 6. Ablations (pick two to run)

1. **Cold-start dosage**: 0 (pure RL) vs 2K vs 6.4K traces
   -> Core contribution: directly answers "does a strong base still need LongVT-style cold start?"
2. **Reward shape**: vanilla IoU vs penalty-aware IoU (extra penalty for severely over-long predicted intervals, suppressing span inflation; prior taken from VideoTemp-o3)
3. (Backup) **Value of multiple turns**: T=3 vs T=1, reported bucketed by video duration
   -> Expected claim: the longer the video and the smaller the target interval's share of it, the larger the relative gain from multiple turns (duration-stratified analysis, a figure absent from the LongVT paper)

---

## 7. Evaluation plan

- **Charades-STA**: R@0.3 / R@0.5 / R@0.7 and mIoU (standard TVG metrics)
- **Long-video held-out** (split from LongVT data): same metrics plus duration buckets
- **Process metrics**: tool call rate, average number of turns, window correction success rate
- **Anchor**: the three zero-shot modes from Step 0; all training results reported as relative improvements

---

## 8. Timeline

| Week | Content |
|---|---|
| Week 1 (in parallel with DUCB-Acc, in spare cycles) | Build the environment; get verl's bundled multi-turn VLM example running; Step 0 probe; trace re-rendering script; start video downloads |
| Week 2 | SFT cold start; finish the evaluation pipeline; finalize the zero-shot baseline |
| Weeks 3–4 (early September, after the summer report is delivered) | Main GRPO experiment + the two ablations |
| Week 5 (mid September) | (Optional) RFT; README / tech report; add to resume |

---

## 9. Risk list

| Risk | Mitigation |
|---|---|
| The full combination of verl LoRA + SGLang + video + tools may be untrodden ground | Reserve a week of debugging budget; when stuck, search the verl issue tracker for `qwen3_vl`; run the official example first in week 1 to validate the chain |
| Qwen3-VL's DeepStack architecture and new processor have a different memory profile than the previous generation | Set the frame cap empirically; start from the 32/16 budget and probe upward |
| A strong base compresses the agentic delta | The Step 0 probe exposes this early; if zero-shot is already strong, shift the narrative to "ablations reveal the marginal value of cold start and multi-turn interaction on a strong base" — a negative result is equally informative |
| A mismatched chat template destroys tool calling | After the re-rendering script is produced, run a small-batch SFT and manually inspect output formatting before going full scale |
| No cross-generation reference numbers | Anchor on our own zero-shot baseline and report everything as relative improvement |

---

## 10. Resume and interview narrative

- **One-liner**: Agentic Temporal Video Grounding with Verifiable Rewards — built on Qwen3-VL-4B + verl, trained with GRPO using multi-turn crop_video tool calls and IoU-verifiable rewards, systematically examining on a single GPU the contribution of cold-start dosage, reward shape, and multi-turn interaction to temporal grounding.
- **Pairing with DUCB-Acc**: one is bandit scheduling on the preference-data side (the DPO line), the other is reward design and multi-turn rollout for agentic RL (the GRPO line); together they cover the two main pillars of LLM post-training roles.
- **How to describe the data strategy**: "the entire engineering budget went into training strategy and reward design; data was taken from open-source releases" — a plus for algorithm-focused post-training roles.
