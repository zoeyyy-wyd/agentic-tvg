# SFT run `sft_mix` — Qwen3-VL-4B + LoRA

Cold-start SFT for the agentic video-QA line (DATA.md). One A100 80GB,
verl 0.9.0 SFT trainer. Finished 2026-08-26 19:51.

## Result

| step | train/loss | val/loss |
|---:|---:|---:|
| 25 | 1.121 | 1.124 |
| 50 | 0.953 | 1.006 |
| 75 | 0.890 | 0.964 |
| 100 | 0.943 | 0.946 |
| **120 (final)** | **0.934** | **0.938** |

train/loss 1.760 -> 0.934 over 120 steps (1,923 rows x 2 epochs, batch 32).
val/loss fell monotonically at every checkpoint and never diverged from
train/loss -- the two end within 0.004 of each other. No overfitting at 2
epochs, and the per-checkpoint gain is halving (-0.119, -0.041, -0.018,
-0.009), so a third epoch would buy very little.

grad_norm 1.567 -> 0.279, MFU steady at ~0.317, peak GPU 39.0G allocated /
42.3G reserved of 80G.

## Files

| file | what |
|---|---|
| `ckpt/global_step_120/` | the checkpoint, 19G — written here directly by the trainer |
| `curves.png` | loss / memory / lr / grad_norm / mfu panels |
| `metrics.csv` | all 120 steps, one row per step — the form that outlives the log |
| `hydra_config.yaml` | fully-resolved config: verl's defaults *and* our overrides |
| `hydra_overrides.yaml` | just the command-line overrides |
| `console.log` | raw merged console output of every attempt |

`ckpt/` and `console.log` are gitignored; the rest is small and committed.

Of the checkpoint's 19G, only **126 MiB** is the LoRA adapter (504 of 1,218
tensors); the rest is the frozen fp32 base, already in
`models/Qwen3-VL-4B-Instruct/`. Export the adapter if this run needs to
outlive the next one — `max_ckpt_to_keep=1` means a rerun of `sft_mix`
deletes `global_step_120`.

## Reproduce

```bash
bash run_sft.sh          # ~2h on one A100 80GB
```

`curves.png`, `metrics.csv` and `console.log` are regenerated on exit by
run_sft.sh's EXIT trap — crashed runs included, since that is exactly when you
want the curve. To redraw by hand:

```bash
python plot_sft.py results/sft-mix/console.log \
    -o results/sft-mix/curves.png --csv results/sft-mix/metrics.csv
```

## Run history

Two attempts; the checkpoint is continuous across them, and `console.log`
concatenates both (repeated steps keep their last value, which is the one that
survived the rollback).

1. **10:18-11:15** — died at step 50/120, `IndexError` in a dataloader worker.
   One training row (312) carried a literal `<image>` inside the final
   assistant `<think>`, echoed from upstream trace `rft_9397`. verl's SFT
   dataset splits every message string on `<image>`/`<video>` regardless of
   role, so the row promised 31 images and shipped 30. Fixed in
   `data_prep/render_traces.py` (scrub at parse time + assert
   placeholders == assets before writing); see DATA.md §0.5 and §7.2.
2. **17:57-19:51** — resumed from `global_step_25`, ran to 120 clean.

One manual intervention during the resume: verl tracks checkpoint retention in
an in-memory list, so a resumed run never deletes the checkpoint it resumed
FROM. `global_step_25` was orphaned and `/` hit 91%; step 75 would have died
with ENOSPC. Deleted by hand. Noted in `run_sft.sh`.
