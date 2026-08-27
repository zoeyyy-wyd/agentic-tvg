# Zero-shot vs SFT on all 114 rl_val rows

Paired: same rows, same config, same judge (Haiku 4.5, temp 0), greedy decoding
(`val_kwargs.do_sample=False`). Only the model differs. 2026-08-26/27.

| metric | base | SFT-merged | delta |
|---|---:|---:|---:|
| format_score | 0.0000 | **0.4868** | +0.4868 |
| answered | 0.3070 | **0.9737** | +0.6667 |
| acc | 0.1447 | **0.4649** | **+0.3202** |
| evidence_iou | 0.1193 | 0.1500 | +0.0307 |
| num_tool_calls | 1.9298 | 0.9737 | −0.9561 |
| **reward** | 0.2044 | **1.0267** | **+0.8223** |

`format_score` is 0 or FORMAT_BONUS=0.5, so 0.4868 means ~97% of rows passed
and 0.0000 means none did. `judge_used` = 0.9737 on the SFT arm, equal to
`answered`: every parsed answer really went to the judge, none fell back to the
alias matcher.

## What it says

**SFT did more than fix the output format.** acc 0.145 -> 0.465 is a 3.2x
improvement in answer correctness, not a formatting artifact. An earlier
10-row comparison put acc at +0.05 and concluded the gain was purely the
output *form*; at n=114 that conclusion is wrong — the 10-row slice was both
too small and easier (base answered .60 there vs .307 here).

The base model was never bad at *calling* the tool: 1.93 crop_video calls per
row, IoU 0.119. It was bad at stopping — it never produced a parseable
`<think>/<answer>`, so `answered` was 0.307. SFT cut tool calls to 0.97 (the
one call its training traces demonstrate) while IoU still rose to 0.150:
fewer calls, better aimed.

## Provenance

- **SFT arm**: step-0 validation of the GRPO run, 2026-08-27 00:34, model
  `results/sft-mix/merged`. Per-row dump preserved here as
  `sft_rollouts.jsonl` (114 rows: input, output, gt, and every reward
  component). It was written to `results/grpo-vanilla/val_rollouts/0.jsonl`,
  which the next GRPO run overwrites — hence the copy.
- **base arm**: full-val pass on 2026-08-26 22:47, model
  `models/Qwen3-VL-4B-Instruct`. Aggregate metrics only; this run predates
  `trainer.validation_data_dir` being set, so there is no per-row dump.

Reproduce either arm:

```bash
MODEL_PATH=$PWD/models/Qwen3-VL-4B-Instruct \
  bash run_grpo.sh trainer.val_only=True EXP_NAME=val_base    # base
bash run_grpo.sh trainer.val_only=True EXP_NAME=val_sft       # SFT (default MODEL_PATH)
```
