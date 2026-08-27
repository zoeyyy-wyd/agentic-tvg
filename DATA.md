# Data Manifest — every row traced to its LongVT source

Companion to `PLAN.md` (why and what) and `prepare_data.sh` (how to fetch).
All numbers measured against the released data on 2026-08-25/26. "The paper"
= LongVT (arXiv:2511.20785).

---

## 0. The pipeline — how every training file is made

```
prepare_data.sh                                  [30-60 min, one command]
  downloads: 11 annotation parquets (163M) · selfqa/rl_val/geminicot videos
             (~22G) · Qwen3-VL-4B (8.3G); resumable, skips populated dirs

        │
        ▼
data_prep/render_traces.py                       [~2 min re-run; first run
  reads : selftrace + selfqa + geminicot          decodes ~31K frames]
  does  : 1. dedup selftrace 15,354 traces → 1,290 questions;
             join selfqa on exact question text (1,157 match)
          2. (judge era, 2026-08-26) NO answer-length cut: R_acc is scored by
             the LLM judge (reward.py/judge.py), so long answers are usable;
             --max-gt-words now defaults 999 (was 6, the matcher-era value)
          3. allocate questions — SFT 600 / RL 1,068, disjoint → allocation.json
          4. per kept trace: rebuild prompts (prompts.py), canonicalize
             tool_call, re-decode 30 crop frames from the mp4 → frames/*.jpg
             (C=30 and F=128 since 2026-08-26; see FRAMES_SWEEP.md)
  writes: sft_train.parquet (1,923) · sft_val.parquet (35, split by video id)
          · allocation.json · frames/ (57,810 jpgs)

        │  (allocation.json carries the RL question list forward)
        ▼
data_prep/extract_rl.py                          [~2 s, no decoding]
  reads : allocation.json + selfqa/rl_val annotations + video roots
  does  : rebuild prompts with the SAME builders (byte-identical to SFT);
          expand each GT into a frozen alias list
          (answer_match.expand_aliases: normalize → parenthetical variants
          → number-word variants); validate video/duration/segment
  writes: rl_train.parquet (1,068) · rl_val.parquet (114)
```

Deterministic end to end (seed 0, no wall-clock inputs): same downloads →
byte-same allocation and rows. Changing GLOBAL_NUM_FRAMES or any prompt
string requires re-running both scripts (the frame count is baked into the
system prompt).

## 1. Our datasets ← LongVT releases

| Our file | Size | Composition | LongVT source (role in the paper) |
|---|---|---|---|
| `sft_train/sft_val.parquet` | 1,958 rows | 1,358 selftrace traces + 600 geminicot traces; GT + evidence-window metadata from selfqa | `rft_selftrace_15k3` (**stage-3 RFT**) + `sft_geminicot_4k8` (**stage-1 cold start**) + `rl_selfqa_1k6` (**stage-2 RL**, metadata only) |
| `rl_train.parquet` | 1,068 rows (judge era; difficulty filtering still expected to trim) | selfqa questions disjoint from SFT — no answer-length cut since the LLM judge scores R_acc; GT still expanded to frozen alias lists (judge-unavailable fallback + audit) | `rl_selfqa_1k6` (**stage-2 RL train**) |
| `rl_val.parquet` | 114 rows | used verbatim | `rl_val_114` (**stage-2 RL val**); zero video overlap with selftrace (verified) |

These first-column files are all **generated locally** by
`data_prep/render_traces.py` / `data_prep/extract_rl.py`; the table states
their provenance, not a download location.

The key deliberate role inversion: **the paper's stage-3 RFT data serves as
our stage-1 cold start.** It is doubly filtered (answer judged correct AND
crop-vs-evidence IoU ≥ 0.3), so it is *stronger* per sample than the paper's
own cold-start mixture — disclosed as a confound in PLAN §3.4. In exchange,
our own stage-3 self-distillation must come from our own rollouts.

## 2. Disposition of all 11 LongVT-Parquet files

| File | Rows | Role in the paper | Here |
|---|---:|---|---|
| `rft_selftrace_15k3` | 15,354 | stage-3 RFT (their model's successful rollouts) | **SFT backbone**: dedup → 1,290 questions; take 600 × ≤3 traces = 1,379 |
| `rl_selfqa_1k6` | 1,668 | stage-2 RL train | **RL backbone** (1,068 questions) + SFT metadata (GT, video_segment) |
| `rl_val_114` | 114 | stage-2 RL val | **eval set**, verbatim |
| `sft_geminicot_4k8` | 4,881 | stage-1 cold start, Gemini-distilled (paper says 12,766; only 4,881 released) | **SFT supplement**: 600 sampled for question diversity |
| `sft_tvg_6k3` | 6,395 | stage-1 cold start, Qwen-distilled pure TVG | unused (the TVG line was removed 2026-08-26; traces stay in the annotations download) |
| `sft_llavacot_54k5` | 54,591 | stage-1 non-tool image CoT | unused (was the cut CoT-mix ablation's pool) |
| `sft_openvlthinker_2k8` | 2,829 | ditto | unused |
| `sft_wemath_602` | 602 | ditto | unused |
| `sft_videor1_165k5` | 165,575 | stage-1 non-tool video CoT | unused (videos: 128 GiB) |
| `sft_longvideoreason_5k2` | 5,238 | ditto | unused (videos: 645 GiB) |
| `sft_longvideoreflection_3k` | 3,004 | tool-augmented traces (**not** plain CoT — 200/200 sampled carry tool_calls) | unused (videos: 253.6 GiB) |

## 3. The SFT allocation, derived

```
selftrace 15,354 traces ─dedup→ 1,290 questions (median 7 traces/question,
    max 87; 5 malformed traces discarded)
    join selfqa's 1,668 questions (exact question-text match)
    ├─ 1,157 present on both sides
    ├─   132 selftrace-only → dropped (videos ship only in the unpurchased
    │        51.5 GiB selftrace archives; ~1.5K traces go with them)
    └─   510 selfqa questions without traces (= their model's total failures)

Second cut — REMOVED 2026-08-26 (judge era). The matcher-era pipeline split
answers at normalized GT ≤ 6 words and forced 240 long-answer questions into
SFT while dropping 175 no-trace long-answer questions. With R_acc scored by
the LLM judge (reward.py/judge.py), every answer is scorable, so:

with traces (1,157)   600 sampled → SFT (weight 1/n_traces, leaning hard),
                      557 → RL
no traces (510)       all → RL
dropped               0

SFT = 600 questions × ≤3 traces each, preferring answer-distinct wordings
    → 1,358 traces
geminicot: 600 rows sampled (seed 0); 2% val split by video id
```

### Why the RL pool is 1,068 when the paper trained on 1,668

The gap is purchased, not suffered: the paper cold-started on *other* data
(geminicot/tvg), leaving its full selfqa intact for RL. We repurposed
selftrace — whose questions ARE selfqa questions — as the cold start, so RL
must forfeit those 600 (training RL on questions whose answers sit in the SFT
set would inflate reward and void the result): **1,068 = 1,668 − 600.**
(Matcher era this was 893 — a further 175 long-answer no-trace questions were
dropped as matcher-unverifiable; the judge made them scorable, 2026-08-26.)

Why the smaller pool still works: RL "data" is only the question — the
learning signal is freshly generated rollouts at each visit. At the current
production candidate (150 steps × 16 prompts, K=16): 2,400 visits ≈ 2.2
epochs over 1,068 (the paper: ~3 epochs over 1,668 — same order; BS=32 would
give 4.5). The real small-pool risk is **saturation**: as the policy masters
questions, groups turn all-correct → zero variance → the effective pool
shrinks during training. Countermeasure: early-stop at reward plateau
(PLAN §6). Recovery knob if it saturates too early:
`--sft-questions 400` re-split returns ~200 questions to RL
(re-render + re-extract, ~30 min) — to be pulled on evidence, not
preemptively.

Row ratio **selftrace : geminicot ≈ 7 : 3** (question ratio 1 : 1 — selftrace
averages 2.3 kept traces per question). All knobs live in
`data_prep/render_traces.py`: `--sft-questions / --traces-per-q /
--geminicot-n / --max-gt-words`.

## 4. Against the paper's SFT recipe

The paper's stage-1 cold start totals 247,996 samples; ours takes:

| Paper component | Rows | Here |
|---|---:|---|
| Video-R1 video CoT | 165,575 | 0 |
| Image CoT (LLaVA / OpenVLThinker / We-Math) | 58,022 | 0 |
| LongVideo-Reason video CoT | 5,238 | 0 |
| Gemini-distilled tool traces | 12,766 (4,881 released) | **600** |
| Qwen-distilled TVG traces | 6,395 | 0 (TVG line removed) |
| — (not in their stage 1) | — | **1,379 selftrace** (their stage-3 data) |
| **Total** | **247,996** | **1,958** (127× smaller) |

The 125× gap is the project's bet, not an oversight: LoRA on a strong instruct
base should not need the 229K general-CoT anti-forgetting filler (that
question was the cut dose ablation; we now follow the paper and cold-start).

## 5. Video/image intake (LongVT-Source: 1,116 GiB upstream)

| Archive | GiB | Feeds |
|---|---:|---|
| `selfqa_1.zip` | 5.6 | SFT (global view + tool-frame re-decoding for selftrace rows) **and** RL rollouts — same files, both stages |
| `rl_val_1.zip` | 0.4 | eval |
| `geminicot_1.zip` + `geminicot_2.zip` | 15.8 | SFT geminicot rows — **deleted 2026-08-27** to make room for GRPO checkpoints. The rendered rows live on in `sft_train.parquet`; re-rendering SFT data needs these back |
| **Subtotal (+ model 8.3G)** | **~30** | |
| Not taken | ~1,090 | incl. `selftrace_1..6` (51.5 — their jpgs are replaced by re-decoding from selfqa mp4s), `tvg_*` (11.9), everything else |

The pre-cropped jpgs referenced by selftrace traces are **not downloaded**: at
render time the trace's crop window is re-decoded from the local mp4 with
`agentic_tvg/video_frames.py` — the same code path the RL tool executes
(zero train/serve skew) — into `data/processed/frames/`.

## 6. Dropped (paper trail for the limitations section)

- 175 questions: answers >6 words and no traces (unverifiable and unimitable)
- 132 questions (~1.5K traces): videos absent from every purchased archive
- ~12K traces: redundant solutions of the same question, removed by dedup
- 5 traces: malformed
- RL is 1,068 after the judge-era rules (the matcher-era answer-length cut
  gave 893). Offline difficulty filtering was considered and dropped: measured
  at the SFT policy's acc 0.465 with K=16, degenerate groups (all-correct or
  all-wrong, hence zero advantage) are 0.005% of the pool — not worth a pass

## 7. What the data actually looks like

Real rows pulled from disk (long fields abridged with `[...]`).

### 7.1 Raw — `longvt_rl_selfqa_1k6.parquet` (one row = one RL question)

```
prompt[-1].content : <video>After the man in a pink shirt and white cap drops his piece
                     of paper at the starting platform, what does he wave to start the
                     kayak race? Think first, call **crop_video** if needed, then [...]
                     The Video path for this video is: -0NUlZvrYY4.mp4
reward_model       : {'ground_truth': 'A red flag.', 'style': 'model'}
extra_info         : {question, answer, video_segment: [10. 20.], need_tools_kwargs,
                      tools_kwargs, index, split}
```

`ground_truth` feeds the answer matcher; `video_segment` feeds R_time. The
"Video path" sentence never survives into our data (see 7.3).

### 7.2 Raw — `longvt_rft_selftrace_15k3.parquet` (one row = one solution trace)

Five messages in the legacy Qwen2.5-VL-era storage format:

```
[0] system    imgs=0  : You are a helpful assistant. # Tools You may call one or more
                        functions [...] (their tool schema, embedded as text)
[1] user      imgs=0  : Which musical instrument appears on the "L'Etoile Gagnante" box
                        [...] Think first, call **crop_video** if needed [...]
                        The Video path for this video is: PfzDzMeVOKo.mp4
[2] assistant imgs=0  : <think>First, I'll examine the initial graphics [...]</think>
                        <tool_call>{"name":"crop_video","arguments":{"video_path":
                        "PfzDzMeVOKo.mp4", "start_time": 0.0, "end_time": 49.0}}</tool_call>
[3] user      imgs=48 : <tool_response> The tool executed successfully. Here are the
                        processed result</tool_response>          ← role=user (!), literal
                        wrapper text, 48 jpg refs we do not download
[4] assistant imgs=0  : <think>[...] from [0.00s – 0.05s] [...]</think>
                        <answer>An acoustic guitar.</answer>
```

Legacy quirks that must not survive: tool response stored as `role=user` with a
literal `<tool_response>` wrapper (the Qwen3 template adds its own), a
model-echoed `video_path` argument, no spaces after `:` in the JSON, frame
counts far over our 16-frame budget (median 30, max 300), and — in exactly one
trace, `rft_9397` — a literal `<image>` inside message [4]'s `<think>`, where
the model echoed a tool header. That last one is invisible upstream and fatal
downstream: the trainer counts it as a real placeholder, so the row promises 31
images and ships 30. Scrubbed at parse time; see PLAN.md #4.5.

### 7.3 Processed — `sft_train.parquet` (one row = one training sample)

```
messages:
[0] system   : You are a precise video question answering assistant. [...]     ← ours,
               rebuilt from prompts.py (QA mode), byte-equal to RL serving
[1] user     : <video>\nThe video is 234.0 seconds long. Question: "According to the
               onscreen titles, which optional-height diving event [...]"
               Answer the question based on the video.            ← no video-path sentence
[2] assistant: <think>Let me think through this step-by-step. [...]</think>
               <tool_call>
               {"name": "crop_video", "arguments": {"start_time": 0.0, "end_time": 161.0}}
               </tool_call>                                        ← canonical spacing,
                                                                     no video_path
[3] tool     : <image>×16 + "The 16 frames above are sampled evenly from 0.0s to 161.0s.
               Their timestamps are: 0.0s, 10.7s, [...] 161.0s."   ← role=tool, our
               crop_response_text, real decoded-frame timestamps
[4] assistant: <think>[...]</think><answer>[...]</answer>          ← verbatim, less any
                                                                     stray <image>/<video>

images : 16 × {image: data/processed/frames/<vid>_<s>_<e>_<i>.jpg,
               max_pixels: 150528, min_pixels: 3136}               ← re-decoded by us
videos : 1 × {video: data/videos/selfqa/0D8e06EOBgg.mp4,
              nframes: 64, max_pixels: 50176, min_pixels: 3136}    ← global view,
                                                                     decoded at train time
tools  : [crop_video JSON schema]                                  ← what the chat
                                                                     template renders
extra_info : {question, video_id, source: selftrace|geminicot,
              tool_window, duration, gt, video_segment}            ← audit side-channel,
                                                                     never trained on
```

Loss mask: only messages [2] and [4] are supervised; everything else (vision
tokens included) is conditioning context.

### 7.4 Generated bookkeeping — `allocation.json`

```
{ config:        {sft_questions: 600, traces_per_q: 3, geminicot_n: 600, ...},
  sft_selftrace: [ {question, vid, gt, segment, n_traces, picked: [trace idxs]} ×600 ],
  sft_geminicot: [ {question, vid} ×600 ],
  rl:            [ {question, vid, gt, segment} ×1068 ],  ← extract_rl.py's input
  dropped:       {selftrace_only_questions: 132, selfqa_unverifiable_no_traces: 175,
                  selftrace_bad_parse: 5, geminicot_bad_parse: 1} }
```

### 7.5 Generated — `rl_train.parquet` / `rl_val.parquet` (one row = one RL question)

```
prompt       : [system, user] — same builders as 7.3, byte-identical; no
               answer anywhere in the row (GT exists only as a grading key)
videos       : [{video: data/videos/selfqa/....mp4, nframes: 64, ...}]
reward_model : {style: "rule", ground_truth: <raw GT text>}   # judge grades it
extra_info   : {question, gt_text, video_segment: [10.0, 20.0], duration,
                need_tools_kwargs, tools_kwargs: {crop_video: {create_kwargs:
                {video_path, duration}}}}          ← mounts the tool per row
agent_name   : "tool_agent"                        ← verl multi-turn loop
```

Grading: every answer goes to the cached temp-0 Anthropic judge with
LongVT's rubric {FULL 1.0, PARTIAL 0.5, INCORRECT 0}; without an API key the
scorer falls back to on-the-fly rule-alias containment (answer_match.py:
normalization, parenthetical variants like "stone (rock)", number words).

Scoring (reward.py::compute_score_qa): R = 0.5·format + R_acc(judge)
+ 0.5·best-IoU(any crop call, video_segment).
