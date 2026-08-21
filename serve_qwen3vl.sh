#!/usr/bin/env bash
# vLLM OpenAI server for the Step-0 probe (probe/step0_probe.py).
# hermes tool parser == what verl's ToolAgentLoop uses for Qwen3-VL.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH=${MODEL_PATH:-models/Qwen3-VL-4B-Instruct}
PORT=${PORT:-8000}

# image budget: 32 global + 3 crops x 16 = 80; one native video unused here but allowed
exec vllm serve "${MODEL_PATH}" \
    --served-model-name qwen3-vl-4b \
    --host 127.0.0.1 --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --limit-mm-per-prompt '{"image": 96, "video": 1}'
