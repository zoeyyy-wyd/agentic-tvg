# Agentic-TVG

Agentic temporal video grounding with verifiable rewards: Qwen3-VL-4B +
verl GRPO with a multi-turn `crop_video` tool, on one A100 80GB.

Docs: `agentic_tvg_plan.md` (design) · `ENVIRONMENT.md` (env `verl`) · `DATA.md` (data)

## Layout

```
agentic_tvg/            core library (installed editable: pip install -e .)
  constants.py            frame/token budget — single source of truth
  span.py  reward.py      answer parsing, temporal IoU, verl reward fns
  prompts.py              system/user prompts (direct | tool_optional | tool_forced)
  video_frames.py         PyAV interval sampling (shared: tool + probe)
  data_schema.py          verl parquet row builder
  crop_video_tool.py      verl BaseTool (the model-callable tool)
crop_video_tool.yaml    verl tool registry (multi_turn.tool_config_path)
download_videos.sh      fetch tvg/selfqa/rl_val video sets
serve_qwen3vl.sh        vLLM OpenAI server for the probe
run_grpo_tvg.sh         GRPO training launcher (plan §5 Stage 2)
data_prep/              extract_rl.py · prepare_charades.py
probe/                  step0_probe.py                      (plan §3)
tests/                  pytest suite (24 tests, no GPU needed)
```

## State (2026-08-20)

Done:
- env `verl` verified + video-backend fix (ENVIRONMENT.md §7); model weights local
- LongVT annotations + tvg/selfqa/rl_val videos downloaded (DATA.md §8)
- RL data built: `data/processed/rl_train.parquet` (1668) / `rl_val.parquet` (114)
- full prompt path smoke-tested through verl RLHFDataset + Qwen3-VL processor
  (1381-token initial prompt on a 222 s video — inside budget)

Next:
1. `bash serve_qwen3vl.sh` then `python probe/step0_probe.py --data
   data/processed/rl_val.parquet --limit 50` — Step-0 probe on the val split
2. Charades-STA: fetch `charades_sta_test.txt`, run the tvg-id contamination
   check (DATA.md §8.3), then `data_prep/prepare_charades.py`
3. Step-0 on Charades + decision rule -> SFT dose (plan §3)
4. `bash run_grpo_tvg.sh` (ablations: `REWARD_FN=compute_score_penalty`,
   `MAX_USER_TURNS=1`); difficulty pre-filter script before the main run
5. `render_traces.py` (SFT re-render, Qwen3-VL hermes) — after Step 0
