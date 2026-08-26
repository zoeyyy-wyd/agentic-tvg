# Agentic Video QA

Multi-turn tool-calling video QA with verifiable rewards: Qwen3-VL-4B + verl
GRPO with a `crop_video` tool, on one A100 80GB. Recipe adapted from LongVT
(arXiv:2511.20785) with its LLM judge replaced by programmatic rewards.

Docs: `PLAN.md` (design + runbook) · `DATA.md` (data provenance) ·
`ENVIRONMENT.md` (conda env `verl`).

## Layout

```
agentic_tvg/              core library (installed editable: pip install -e .)
  constants.py              frame/token budget — single source of truth
  prompts.py                system/user prompt builders
  span.py  reward.py        answer parsing, temporal IoU, verl reward fns
  video_frames.py           PyAV interval sampling (shared: tool + probe + render)
  crop_video_tool.py        verl BaseTool (the model-callable tool)
  sft_dataset.py            Qwen3-VL fixes over verl's MultiTurnSFTDataset
prepare_data.sh           one command: downloads + renders the QA training data
run_sft.sh                the one SFT script (SMOKE=1 for a 2-step smoke)
run_grpo.sh           GRPO launcher (QA rework pending, PLAN §8)
serve_qwen3vl.sh          vLLM server for the probe
data_prep/                render_traces.py · inspect_splits.py
probe/  tests/  env_setup/
```

## State (2026-08-26)

- env `verl` verified (torchcodec + LD_PRELOAD fix automated; `env_setup/`)
- QA data pipeline built and smoke-tested end to end:
  `prepare_data.sh` → ~1,979 SFT rows → `SMOKE=1 bash run_sft.sh` passed
- Next: full SFT (PLAN §9 Step 3), then the RL-side work list (PLAN §8)
