# Data Manifest — every row traced to its LongVT source

Companion to `PLAN.md` (why and what) and `prepare_data.sh` (how to fetch).
All numbers measured against the released data on 2026-08-25/26. "The paper"
= LongVT (arXiv:2511.20785).

---

## 1. Our datasets ← LongVT releases

| Our file | Size | Composition | LongVT source (role in the paper) |
|---|---|---|---|
| `sft_train/sft_val.parquet` | ~1.9K rows | 1,379 selftrace traces + 600 geminicot traces; GT + evidence-window metadata from selfqa | `rft_selftrace_15k3` (**stage-3 RFT**) + `sft_geminicot_4k8` (**stage-1 cold start**) + `rl_selfqa_1k6` (**stage-2 RL**, metadata only) |
| RL parquet (pending, PLAN §8) | 893 questions (500–700 after difficulty filtering) | selfqa questions disjoint from SFT, answers ≤6 words, plus offline alias sets | `rl_selfqa_1k6` (**stage-2 RL train**) |
| Eval set | 114 questions | used verbatim | `rl_val_114` (**stage-2 RL val**); zero video overlap with selftrace (verified) |

These first-column files are all **generated locally** by
`data_prep/render_traces.py` / the pending `extract_rl.py`; the table states
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
| `rl_selfqa_1k6` | 1,668 | stage-2 RL train | **RL backbone** (893 questions) + SFT metadata (GT, video_segment) |
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

Second cut, by answer verifiability (normalized GT ≤ 6 words):
                 verifiable                unverifiable
with traces      918: 360→SFT / 558→RL    240 → all to SFT (matcher-unusable)
no traces        335: all→RL              175 → dropped

SFT = 240 forced + 360 sampled (weight 1/n_traces, leaning hard) = 600 questions
    × ≤3 traces each, preferring answer-distinct wordings → 1,379 rows
geminicot: 600 rows sampled (seed 0) → SFT total ~1,979 rows, 2% val by video id
```

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
| **Total** | **247,996** | **~1,979** (125× smaller) |

The 125× gap is the project's bet, not an oversight: LoRA on a strong instruct
base should not need the 229K general-CoT anti-forgetting filler (that
question was the cut dose ablation; we now follow the paper and cold-start).

## 5. Video/image intake (LongVT-Source: 1,116 GiB upstream)

| Archive | GiB | Feeds |
|---|---:|---|
| `selfqa_1.zip` | 5.6 | SFT (global view + tool-frame re-decoding for selftrace rows) **and** RL rollouts — same files, both stages |
| `rl_val_1.zip` | 0.4 | eval |
| `geminicot_1.zip` + `geminicot_2.zip` | 15.8 | SFT geminicot rows |
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
- RL's 893 is an upper bound: offline difficulty filtering (K=8 rollouts,
  drop all-correct/all-wrong groups) is expected to keep 500–700
