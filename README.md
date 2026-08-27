# Agentic Video QA

Multi-turn tool-calling video QA with verifiable rewards: Qwen3-VL-4B + verl
GRPO with a `crop_video` tool, on one A100 80GB. Recipe adapted from LongVT
(arXiv:2511.20785), with the paper's LLM judge kept for R_acc and a temporal
IoU term added for evidence grounding.

Docs: `PLAN.md` (design + runbook) · `DATA.md` (data provenance) ·
`GRPO_NOTES.md` (why the RL config is what it is) ·
`env_setup/ENVIRONMENT.md` (conda env `verl`) · `results/*/README.md` (runs).

## Layout

```
agentic_tvg/              core library (pip install -e .)
  constants.py              frame/token budget — single source of truth
  prompts.py                system/user prompt builders
  span.py  answer_match.py  answer parsing, temporal IoU, GT alias expansion
  judge.py  reward.py       LongVT-rubric judge + the verl reward functions
  video_frames.py           PyAV interval sampling (shared: tool + render)
  crop_video_tool.py        verl BaseTool (the model-callable tool)
  sft_dataset.py            Qwen3-VL fixes over verl's MultiTurnSFTDataset
prepare_data.sh           downloads + renders all training data
data_prep/                render_traces.py (SFT) · extract_rl.py (RL)
run_sft.sh                SFT (SMOKE=1 for a 2-step smoke)
export_adapter.py         SFT checkpoint -> merged HF model for GRPO to start from
run_grpo.sh               GRPO (trainer.val_only=True turns it into the evaluator)
plot_sft.py plot_grpo.py  console log -> curves.png + metrics.csv
results/<run>/            ckpt/ + curves + metrics + config snapshot per run
tests/  env_setup/
```

## State (2026-08-27)

| Stage | Status |
|---|---|
| Data | 1,958 SFT rows · 1,068 RL train · 114 RL val |
| SFT | **done** — val/loss 1.124 → 0.938, `results/sft-mix/` |
| Eval | **done** — n=114 paired, `results/eval-114/`: acc 0.145 → 0.465, reward 0.204 → 1.027 |
| GRPO | config frozen and smoke-tested (3 steps, peak RAM 132/188 G); the 267-step run is next |

GRPO needs no separate eval script: `bash run_grpo.sh trainer.val_only=True`
runs `_validate()` and exits, so evaluation and RL share one code path and
cannot drift apart.
