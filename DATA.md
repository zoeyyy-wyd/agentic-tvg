# Data Plan — allocation and disk budget

What to download, what not to, and why. Companion to `ENVIRONMENT.md`.

Upstream total is **1116 GiB**. This project's working set is **~31 GB**. The split is done for us by the upstream repo layout — no batching or delete-as-you-go needed.

Annotations: `longvideotool/LongVT-Parquet` (170 MB total, all splits).
Videos: `longvideotool/LongVT-Source` (123 zip archives, grouped by source subset).

---

## 1. Upstream inventory

Verified against the HF tree API on 2026-08-19.

| Archive group | n | GiB | Used? |
|---|---:|---:|---|
| longvideoreason | 66 | 644.6 | no |
| longvideoreflection | 27 | 253.6 | no |
| videor1 | 13 | 128.1 | no |
| selftrace | 6 | 51.5 | deferred (Stage 3 control) |
| geminicot | 2 | 15.8 | no |
| **tvg** | **2** | **11.9** | **yes** |
| **selfqa** | **1** | **5.6** | **yes** (pending §6.1) |
| llavacot | 1 | 4.6 | no |
| **rl_val** | **1** | **0.4** | **yes** |
| openvlthinker | 1 | 0.07 | no |
| wemath | 1 | 0.01 | no |
| | **123** | **1116.1** | |

The excluded 1063 GiB is LongVT's general long-video QA corpus — plan §4.1 rules it out as irrelevant to pure TVG.

---

## 2. Allocation by stage

Parquet filenames map 1:1 onto archive names, which is what makes the subsetting clean.

| Stage | Annotation (LongVT-Parquet) | Items | Video archive (LongVT-Source) | GiB |
|---|---|---:|---|---:|
| SFT cold start | `longvt_sft_tvg_6k3.parquet` | 6,395 | `tvg_1.zip`, `tvg_2.zip` | 11.9 |
| GRPO train | `longvt_rl_selfqa_1k6.parquet` | ~1,600 | `selfqa_1.zip` | 5.6 |
| GRPO val | `longvt_rl_val_114.parquet` | 114 | `rl_val_1.zip` | 0.4 |
| Eval (short) | Charades-STA (AllenAI, separate) | 3,720 test | `Charades_v1_480.zip` | ~13 |
| Eval (long) | held-out split off `tvg` | — | (already local) | 0 |
| RFT (Stage 3) | our own RL rollouts, IoU >= 0.5 | — | (already local) | 0 |
| RFT control (optional) | `longvt_rft_selftrace_15k3.parquet` | 15,300 | `selftrace_1..6.zip` | 51.5 |

Per-archive detail for what we take:

```
tvg_1.zip      10.01 GiB
tvg_2.zip       1.92 GiB
selfqa_1.zip    5.55 GiB
rl_val_1.zip    0.37 GiB
```

**Stage 3 note**: plan §5 Stage 3 is self-distillation from our own rollouts, so it needs no new video — it reuses `tvg`/`selfqa` videos already on disk. The upstream `rft` split is a *different* thing: a ready-made 15.3K trace set that would serve as a free comparison arm. Deferred, not cancelled; 51.5 GiB only if Stage 3 actually happens and there is room.

---

## 3. Disk budget

`/` has 89 GB free (148 GB total, `/dev/sda1`, the only volume).

| Item | GB |
|---|---:|
| conda env `verl` | 12 (already spent) |
| Qwen3-VL-4B-Instruct weights | 9 |
| LongVT annotations (all splits) | 0.2 |
| tvg + selfqa + rl_val video | 18 |
| Charades-STA v1_480 | ~13 |
| unzip headroom (peak, before deleting zips) | ~18 |
| **subtotal** | **~58** |
| remaining for checkpoints / rollout logs | **~31** |

Peak is during unzip, when zip and extracted tree coexist. Delete each zip right after extracting and the steady state drops to ~40 GB used.

---

## 4. Download

```bash
# annotations — all splits, 170 MB, download in full
hf download longvideotool/LongVT-Parquet --repo-type dataset \
    --local-dir data/annotations

# video — only the three subsets we need
hf download longvideotool/LongVT-Source --repo-type dataset \
    --include "tvg_*.zip" "selfqa_1.zip" "rl_val_1.zip" \
    --local-dir data/archives
```

`--include` is what keeps this at 18 GiB instead of 1.1 TiB. Do not run a bare `hf download` on LongVT-Source.

Charades-STA: video package and annotations come from AllenAI directly, not HuggingFace.

---

## 5. Layout

```
data/
  annotations/          # LongVT-Parquet, verbatim
  archives/             # downloaded zips, deleted after extract
  videos/
    tvg/                # <- tvg_1,2.zip
    selfqa/             # <- selfqa_1.zip
    rl_val/             # <- rl_val_1.zip
    charades/           # <- Charades_v1_480.zip
  processed/
    sft_traces.jsonl    # render_traces.py output, Qwen3-VL template
    rl_train.parquet    # extract_rl.py output, (video, query, gt_span)
    rl_val.parquet
    charades_test.json
```

Media paths in the parquet files are sanitized upstream and must be rewritten to local paths — that rewrite belongs in `render_traces.py` / `extract_rl.py`, not a separate pass.

---

## 6. Open items

### 6.1 Does the RL split actually carry time-span GT? (blocking for `extract_rl.py`)

Plan §4.1 assumes the RL split is `(video, query, GT window)` triples. But the split is named **selfqa** — self-generated QA, which is plausibly open-ended text answers rather than time intervals. If there is no interval GT, IoU reward is not computable on it and the RL data must instead be carved out of the 6,395 `tvg` items (disjoint from the SFT portion), which changes plan §4.1.

Resolve by downloading `longvt_rl_selfqa_1k6.parquet` (tens of MB) and inspecting the schema. Do this before writing any data script.

### 6.2 SFT/RL overlap

If 6.1 forces RL data to come from `tvg`, the 6,395 items must be split disjointly — and the "lightweight SFT = 2K traces" branch of the Step 0 decision rule (plan §3) then leaves ~4K for RL, which is comfortable. The "full SFT = 6.4K" branch does not, and would need the split renegotiated. Note this when Step 0 lands.

### 6.3 Decode cost vs. a frame cache

`crop_video` re-decodes an arbitrary interval on every tool call. For long videos this may dominate rollout wall-clock. If it does, pre-decode each video to a low-fps low-res JPEG cache and have the tool sample from that. Deferred — measure during the Step 0 probe before building it.

### 6.4 Charades package size unverified

~13 GB is from memory, not checked against the AllenAI server. Confirm before committing the disk budget in §3.

---

## 7. Plan amendments

Supersedes `agentic_tvg_plan.md`:

- **§4.3** reserves 150–200 GB for video. Actual requirement is **~31 GB**. The reserve was estimated against the full corpus, not this project's subset.
- **§4.1** lists RL data as "~1.6K, topped up to 3–5K from the grounding subset". The top-up may become mandatory rather than optional — see §6.1.
- **§4.1** excludes "LongVT's 220K non-tool CoT items". Concretely those are the `longvideoreason` / `longvideoreflection` / `videor1` / `geminicot` / `llavacot` groups, 1047 GiB of the 1116.

`ENVIRONMENT.md` §5's disk open item ("locate a larger mount, or download in batches and delete as you go") is resolved — neither is needed.

---

## 8. Resolutions (2026-08-20, after download + inspection)

### 8.1 §6.1 resolved — RL split carries span GT after all

`longvt_rl_selfqa_1k6` / `longvt_rl_val_114` have `reward_model.ground_truth`
as free text (style="model") — but **every** row carries
`extra_info.video_segment = [start, end]`, the QA evidence window. 1668/1668
and 114/114 rows validated against the actual videos with zero drops.
`data_prep/extract_rl.py` recasts each row as grounding (query := question,
GT := video_segment) -> `data/processed/rl_train.parquet` (1668) and
`rl_val.parquet` (114). Durations: train 149–301 s (median 205), val
150–301 s (median 218). §6.2 is therefore moot — RL data does not touch the
`tvg` items, and both Step-0-decision branches keep all 6,395 traces for SFT.

### 8.2 The tvg split is not long-video

All 6,395 SFT traces are `tvg_charades_cot_*`. The "eval (long) held-out off
`tvg`" row in §2 is wrong — the long-video eval role is served by
`longvt_rl_val_114` instead (150–301 s videos, disjoint from training).

### 8.3 tvg archive contents

Flat directory: 2,454 mp4 (1,859 with 5-char Charades ids, 595 others) plus
60,616 pre-cropped trace frames named `{vid}_{start}_{end}_{idx}.jpg`. Two
consequences for `render_traces.py` (SFT stage): tool-response frames can be
re-used from these jpgs without decoding, and the 1,859 Charades ids must be
checked for overlap against the Charades-STA **test** ids once
`charades_sta_test.txt` is in — any intersection is eval contamination and
also decides whether `Charades_v1_480.zip` (~13 GB) still needs downloading.

### 8.4 §6.3 partially resolved

qwen-vl-utils now decodes through torchcodec (seek-based, ENVIRONMENT.md §7),
which removes the full-file-read worst case for the *global* view. The crop
tool decodes with PyAV seek+scan; measure during Step 0 as planned.
