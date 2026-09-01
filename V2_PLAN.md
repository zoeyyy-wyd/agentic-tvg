# V2 Plan — SUPERSEDED 2026-09-01, same day it was drafted

The plan this file carried — **SFT2 with `sft_longvideoreflection_3k`'s
2-crop traces injected, then GRPO2 from that model** — was abandoned before
any of it ran, on measurements made the same day. Round 2 as actually
defined (same SFT as v1; judge v2 + epoch-boundary curriculum + constant
lr) lives in `GRPO2_PLAN.md`. This file remains as the post-mortem: what
was measured, and why injection lost.

## 1. What the reflection file turned out to be

`sft_longvideoreflection_3k` (3,004 rows) is **absent from the LongVT
paper's Table 1** — every other released parquet maps onto a table line and
228,835 + 19,161 = 247,996 closes without it. It is most plausibly the
unlisted multi-round portion of the 12,766 Gemini-distilled iMCoTT
(4,881 single-round released as `geminicot`; 4,881 + 3,004 = 7,885 of
12,766): same system prompt / tool schema / "The video path…" user suffix,
zero video and zero question overlap with geminicot, id family
`longvideo_sft_4k_*`. The paper's §3.2 length-adaptive multi-round formula
is visible in the data — 2-crop share rises 36.5% → 100% as video length
crosses ~1,200 s, and geminicot's videos are median ~89 s vs reflection's
~788 s: the two files are the short- and long-video halves of one pipeline.

Structure: 1,367 rows single-crop (5 messages) + 1,637 two-crop
(7 messages); tool responses come back as `user`-role messages whose text
carries no timestamps (frames as image_url entries, ~1 fps, jpgs in
archives we don't download). 642 videos; the 2-crop subset spans 330
videos / 1,562 parseable rows.

## 2. Why injection lost — three measurements

**(a) The traces are better than first sampled, but presuppose a different
budget.** Contrary to the first-draft caveat ("teach the retry format
only"), most 2-crop traces are genuine local refinement: 79.6% of window
pairs overlap, crop2 is typically *wider* (median 24 s vs crop1's 7 s), 69%
of second thinks cite a timestamp seen inside crop1 and anchor crop2 within
±15 s of it; far jumps (gap > 60 s) are 5.5%. The disqualifier is the
**first** crop: median 0.8% of the video — a 124× narrowing (only 9/1,562
start coarse) on videos of median ~944 s, where our F=128 global view is
one frame per ~7.4 s. The generator had per-segment captions; the model at
inference has ~1 global frame inside that window. Training on this
supervises **confidently pinpointing without evidence** — direct downward
pressure on the iou line.

**(b) The RL pool cannot pay for multi-crop.** The selfqa/rl_val videos
(HACS 1,498 + Ego4D-NaQ 170) are 3 fps transcodes of 447/615/903 frames —
**≈149/205/301 s, hard-capped ~302 s** (max GT-window end 302 s; zero rows
beyond 600 s). At 302 s, F=128 gives ~10 in-window global frames (median;
FRAMES_SWEEP) — the first crop is usually right, a second crop buys no IoU
and costs tokens, and round 1's extinction of multi-crop (22 sampled
attempts, mean score below group mean, 20 → 2 over the run) was the reward
working correctly, not a prior-mass accident. An SFT-injected behaviour
would be re-extinguished for the same reason.

**(c) The scope decision.** VideoSIAH-Eval (the shipping benchmark)
averages 1,688 s — reflection's regime, not ours. A crop must be narrower
than 1/4.27 of the video to beat the global view's density (128/30), so
long-video capability at F=128 requires learned hierarchical descent
(≈4.27× narrowing per call, ~78× over 3 calls) — a behaviour no released
file teaches (the reflection traces don't: see (a)). F=512 was measured
infeasible on this box (FRAMES_SWEEP: >180 GB host RAM / >30 min/step).
Decision: **this project is a ≤~300 s-video system**, matching its RL data;
VideoSIAH-Eval is reported as an out-of-budget transfer number with the
arithmetic above disclosed (GRPO2_PLAN §4.6).

Incidental but recorded: video-id overlap reflection × selfqa and × rl_val
is **zero** (measured 2026-09-01), so the injection would have been
leak-clean; the fetch plan (byte-range from ZIP_STORED entries, ~10 GiB /
70 videos / 456 traces) was sound; `data/processed/reflection_video_map.json`
was lost in the 2026-09-01 machine wipe and would need re-deriving from the
Hub listing if this is ever revisited. `render_traces.py` remains
single-window — the N-crop extension was never written.

## 3. What replaced it

`GRPO2_PLAN.md` (rewritten 2026-09-01): SFT stays v1 (`sft_train.parquet`
untouched, `results/sft-mix/merged` the start model), and round 2 changes
the incentive/curriculum side only — judge v2 (the one real reward change),
`compute_score_qa2` (TIME_WEIGHT 1.0, format bonus kept, no shaping term),
constant lr as hygiene, and the epoch-boundary curriculum against pool
saturation. Attribution is recipe-level by declared choice.
