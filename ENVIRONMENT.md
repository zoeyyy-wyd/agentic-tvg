# Environment Setup — conda env `verl`

Target: verl 0.9.0 + Qwen3-VL-4B + multi-turn tool-calling RL on one A100 80GB (srv4-lg2).

Machine: A100 80GB PCIe, **sm80** | driver 610.43.02 / CUDA UMD 13.3 | nvcc 13.3 | Ubuntu 22.04, gcc 11.4 | 64 cores / 188 GB RAM | conda at `/opt/miniconda3` (base is Python 3.14 — do not use).

---

## 1. Version lock

| Component | Version |
|---|---|
| Python | 3.12 |
| CUDA wheel variant | cu130 |
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0+cu130 / 2.11.0+cu130 |
| vllm | 0.24.0 |
| flash-attn | 2.8.3 (FA2) |
| verl | 0.9.0, PyPI install with `--no-deps` |
| transformers | 5.10.4 |
| tensordict | 0.10.0 |
| ray[default] | >=2.41.0 |
| flashinfer-python / -cubin | 0.6.12 (pulled in by vllm) |
| qwen-vl-utils | 0.0.14 (uses PyAV; decord not needed) |
| ffmpeg | via conda-forge |
| numpy / pyarrow | >=2.0.0 / >=19.0.0 |

Binding constraints, in the order they force each other:

- A100 is sm80 -> **FlashAttention 2 only**; FA3/FA4 kernels do not run. This rules out SGLang >=0.5.12, which declares `flash-attn-4` as a hard dependency. Rollout engine is vLLM.
- vllm ships precompiled extensions linked against one exact torch ABI -> `torch==2.11.0` is not negotiable for vllm 0.20–0.26.
- verl 0.9.0 constrains `transformers >=5.5.3, <5.11, !=5.6.0` -> 5.10.4 is the highest usable version.
- vllm 0.24.0 is what verl 0.9.0's CI image (`verlai/verl:vllm024.dev2`) is built on.
- Driver 610 is backward compatible with all CUDA 12.x/13.x wheels; cu130 is chosen to match verl 0.9.0's official Dockerfile.

---

## 2. Install

```bash
conda env remove -n verl        # the existing env is an empty stub (24K, no bin/)
bash scripts/setup_verl_env.sh
```

The script's ordering is load-bearing:

1. `conda create -n verl python=3.12` — base is 3.14; vllm caps at `<3.15` but publishes no 3.14 wheels.
2. Install torch/vision/audio from the cu130 index **first**. pip then treats `torch==2.11.0` as satisfied by `2.11.0+cu130` (PEP 440 local-version rule) and will not swap in the default PyPI variant.
3. Install vllm 0.24.0.
4. Install flash-attn — prebuilt wheel first, source build as fallback (see §3).
5. `pip install -r requirements.txt` — transformers and the rest of verl's dependencies, **after** vllm. vllm's `transformers>=5.5.3` is a lower bound only, so nothing gets clobbered. Everything pip can install normally lives in `requirements.txt`; torch, vllm, flash-attn and verl cannot (per-package index, conditional fallback, `--no-deps`) and stay in the script.
6. `pip install --no-deps verl==0.9.0` — **`--no-deps` is mandatory**; without it pip re-resolves and can replace torch/vllm. (If verl internals ever need patching, switch to a `git clone -b v0.9.0` + `pip install --no-deps -e` editable install.)
7. Run the verification script.

---

## 3. flash-attn

Mandatory, not optional: with `use_remove_padding=True`, verl calls `from flash_attn.bert_padding import ...` in `verl/utils/attention_utils.py` with no CUDA fallback.

Official FA2 wheels stop at torch 2.10, so torch 2.11 needs one of:

- **Prebuilt wheel (default, verified reachable, 231 MB)**
  `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`
- **Source build (fallback, what verl's own Dockerfile does)**
  `MAX_JOBS=32 pip install flash-attn==2.8.3 --no-build-isolation` — 40–90 min. Do not use `MAX_JOBS=64`; the build will exhaust memory.

The setup script tries the wheel and falls back automatically.

---

## 4. Verify

```bash
conda activate verl
python scripts/check_env.py
```

Checks torch CUDA + sm80 detection, a real flash-attn bf16 forward pass, vllm version, `Qwen3VLForConditionalGeneration` import, verl's `qwen3_vl` patch / `ToolAgentLoop` / tool registry, and ray/tensordict/pyarrow/numpy bounds.

Then validate the full path against verl's bundled multi-turn VLM example (geo3k agent-loop) before touching project code.

---

## 5. Open items

- **Disk**: resolved, see `DATA.md` §3. `/` has 89 GB free and is the only volume; the env came in at 12 GB (not 25–30), and the video working set is ~31 GB, not the 150–200 GB the plan reserved. Neither a larger mount nor batched download is needed.
- **SFT framework**: LLaMA-Factory 0.9.5 requires `transformers <=5.6.0`, whose intersection with verl's range is only 5.5.3/5.5.4 — sharing an env would pin transformers five minors behind. Use verl's built-in FSDP SFT trainer, or give LLaMA-Factory a separate env (~25 GB more disk).
- **Plan amendment**: `agentic_tvg_plan.md` §2 specifies SGLang as the rollout engine; it must read vLLM 0.24.0. Multi-turn tool calling in verl 0.9.0 lives in `verl/experimental/agent_loop/tool_agent_loop.py` + `verl/tools/`, which is decoupled from the rollout backend, so nothing else in the plan changes.

---

## 6. Fallbacks

- vllm-side API errors against transformers 5.10.4 -> drop to 5.9.0 or 5.8.1, both still inside verl's allowed range.
- Do not install verl's `[sglang]` or `[trtllm]` extras; they will break this environment.

---

## 7. Video reader backend (resolved 2026-08-20)

The §1 note "qwen-vl-utils uses PyAV" was wrong: qwen-vl-utils 0.0.14 has no
PyAV backend. Its priority is torchcodec > decord > torchvision, and with
neither of the first two installed it falls through to torchvision — whose
`io.read_video` was **removed** in torchvision 0.26, so every video load in
verl's RLHFDataset/agent-loop path crashes.

Fix in place (both parts required):

1. `pip install torchcodec` (now in `requirements.txt`). decord is not an
   option: no Python 3.12 wheels.
2. `conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`
   (already set on this env). torchcodec's ffmpeg-8 core links the conda
   ffmpeg, whose libopenvino needs `CXXABI_1.3.15`; without the preload the
   process binds the system gcc-11 libstdc++ first and the dlopen fails.
   libstdc++ is strictly backward compatible, so preloading the newer one is
   safe — verified torch 2.11 CUDA + vllm 0.24 still import and see the GPU.

Verified end to end: RLHFDataset -> process_vision_info (torchcodec) ->
hermes chat template with the crop_video schema -> Qwen3-VL processor gives
a 1381-token initial prompt for a 222 s rl_val video (32 frames, grid
[16,12,16] = 784 video tokens), inside the 4096 prompt budget.

Note for a future srv3 replica: `agentic-tvg` repo is installed editable
(`pip install -e .` from the repo root) so verl can import
`agentic_tvg.crop_video_tool` and the custom reward file.
