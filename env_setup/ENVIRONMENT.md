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
| qwen-vl-utils | 0.0.14 (video backend = torchcodec, see §7) |
| torchcodec | 0.16.0 (mandatory; needs the §7 LD_PRELOAD) |
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
bash env_setup/setup_verl_env.sh
```

The script's ordering is load-bearing:

1. `conda create -n verl python=3.12` — base is 3.14; vllm caps at `<3.15` but publishes no 3.14 wheels.
2. Install torch/vision/audio from the cu130 index **first**. pip then treats `torch==2.11.0` as satisfied by `2.11.0+cu130` (PEP 440 local-version rule) and will not swap in the default PyPI variant.
3. Install vllm 0.24.0.
4. Install flash-attn — prebuilt wheel first, source build as fallback (see §3).
5. `pip install -r requirements.txt` — transformers and the rest of verl's dependencies, **after** vllm. vllm's `transformers>=5.5.3` is a lower bound only, so nothing gets clobbered. Everything pip can install normally lives in `requirements.txt`; torch, vllm, flash-attn and verl cannot (per-package index, conditional fallback, `--no-deps`) and stay in the script.
6. `pip install --no-deps verl==0.9.0` — **`--no-deps` is mandatory**; without it pip re-resolves and can replace torch/vllm. (If verl internals ever need patching, switch to a `git clone -b v0.9.0` + `pip install --no-deps -e` editable install.)
7. `conda env config vars set LD_PRELOAD=...libstdc++.so.6` — required for
   torchcodec to decode anything (§7). The script also exports it inline so
   step 8 sees it without a reactivate.
8. Run the verification script.

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
python env_setup/check_env.py
```

Checks torch CUDA + sm80 detection, a real flash-attn bf16 forward pass, vllm version, `Qwen3VLForConditionalGeneration` import, verl's `qwen3_vl` patch / `ToolAgentLoop` / tool registry, ray/tensordict/pyarrow/numpy bounds,
and a real torchcodec decode of a synthetic clip + the `LD_PRELOAD` / backend-selection asserts (§7).

Then validate the full path against verl's bundled multi-turn VLM example (geo3k agent-loop) before touching project code.

---

## 5. Open items

- **Disk**: `/` is the only volume, 148 GB total. Downloads and the env fit comfortably (env 12 GB, videos ~31 GB), but **checkpoints do not**: each SFT/GRPO checkpoint is ~19 GB and verl saves the new one before deleting the old, so any run needs 2x that free. Keep `max_ckpt_to_keep=1` set on both trainers, and after a resume delete the checkpoint you resumed from by hand — verl tracks retention in memory and will never reclaim it. As of 2026-08-27, 45 GB free after deleting `data/images` and the geminicot videos.
- **SFT framework**: LLaMA-Factory 0.9.5 requires `transformers <=5.6.0`, whose intersection with verl's range is only 5.5.3/5.5.4 — sharing an env would pin transformers five minors behind. Use verl's built-in FSDP SFT trainer, or give LLaMA-Factory a separate env (~25 GB more disk).
- **Rollout engine**: the original plan specified SGLang; the working choice is vLLM 0.24.0. Multi-turn tool calling in verl 0.9.0 lives in `verl/experimental/agent_loop/tool_agent_loop.py` + `verl/tools/`, which is decoupled from the rollout backend, so nothing else in the plan changes.

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

Fix, both parts required and both now automated by
`env_setup/setup_verl_env.sh` (torchcodec in `requirements.txt` step 4,
LD_PRELOAD in step 7):

1. `pip install torchcodec` — decord is not an option: no Python 3.12 wheels.
2. `conda env config vars set LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`.
   torchcodec dlopens the core matching the installed ffmpeg major (conda-forge
   ffmpeg 8.0.1 -> `libtorchcodec_core8.so`), which pulls in the conda ffmpeg's
   libopenvino and needs `CXXABI_1.3.15`; without the preload the process binds
   the system gcc-11 libstdc++ first and the dlopen fails with

       OSError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15'
       not found (required by .../libopenvino.so.2541)

   **The failure is lazy**: `import torchcodec` succeeds without the preload;
   only the first actual decode raises. `check_env.py`'s `video stack` check
   therefore does a real synthetic-video decode, not an import.
   libstdc++ is strictly backward compatible, so preloading the newer one is
   safe — verified torch 2.11 CUDA + vllm 0.24 still import and see the GPU.

Verified end to end: RLHFDataset -> process_vision_info (torchcodec) ->
hermes chat template with the crop_video schema -> Qwen3-VL processor gives
a 1381-token initial prompt for a 222 s rl_val video (32 frames, grid
[16,12,16] = 784 video tokens), inside the 4096 prompt budget.

Note for a future srv3 replica: `agentic-tvg` repo is installed editable
(`pip install -e .` from the repo root) so verl can import
`agentic_tvg.crop_video_tool` and the custom reward file.

---

## 8. GRPO launch failures on srv1-lg2 (all diagnosed 2026-08-26)

Three independent blockers, hit in sequence while bringing up the first GRPO
smoke. §0's "srv4-lg2" is stale — this replica is **srv1-lg2** and has its own
machine-level quirk (8.1).

### 8.1 Empty `/etc/hosts` → every Ray component times out at startup

**Symptom**: `ray.init()` / `ray start --head` dies with "The current node
timed out during startup … raylet failed to startup or the GCS has become
overloaded"; `raylet.err` shows `Timed out waiting for file …
metrics_agent_port_…`; `dashboard.log` shows `Module MetricsHead failed to
start. Timeout after 30.0 seconds`. Retrying stacks half-dead sessions under
`/tmp/ray` and makes the next attempt worse.

**Root cause**: `/etc/hosts` on srv1-lg2 is **empty** — no `127.0.0.1
localhost` line. Every `localhost` lookup falls through nss `files` to DNS,
where glibc (default `ndots:1`) first appends all five `search` domains from
`resolv.conf` (`localhost.bed.cosmos-lab.org` … `localhost.rutgers.edu`); the
upstream drops the first attempt of each (~5 s timeout + ~1.1 s retry) →
**~28 s per `localhost` resolution**. Reproduce with `time getent hosts
localhost`; `import ray` alone pays it once, inside `ray._private.parameter`.
Ray's GCS / raylet / dashboard / agents each resolve `localhost` repeatedly
under 25–30 s startup deadlines → cascading timeouts. torchrun/SFT survived
because it resolves once, off any deadline.

**Real fix (needs root)**:
`echo "127.0.0.1 localhost" | sudo tee -a /etc/hosts` (and `::1 localhost`).

**Workaround (automated)**: `env_setup/preflight.sh` now exports
`RES_OPTIONS="ndots:0 timeout:1 attempts:1"` when `/etc/hosts` lacks
`localhost`. `ndots:0` tries the bare name first, which systemd-resolved
answers locally in <10 ms (it synthesizes `localhost` itself). Verified:
`import ray` 28.8 s → 0.34 s; Ray head up in ~6 s. Manual `python`/`ray`
invocations outside the run scripts need the same export.

Cleanup after a timeout loop: `ray stop --force`, then delete stale
`/tmp/ray/session_*` dirs. (Careful: `pkill -f gcs_server` matches your own
shell's command line.)

### 8.2 verl 0.9.0 V1 trainer needs `transfer_queue`, which is not installed

`trainer.use_v1` defaults to `true`, and `TaskRunnerV1.run()` does
`import transfer_queue` → `ModuleNotFoundError`. The verl 0.9.0 wheel neither
bundles the module nor declares the dep (it lives on PyPI as `transferqueue`;
unvetted against this env's pins, and the `--no-deps` install philosophy says
don't pull it blind). `run_grpo.sh` pins `trainer.use_v1=False` — the legacy
`main_ppo_v0.TaskRunner` agent-loop path, the one the script's keys were
verified against. The deprecation warning at launch is expected. If we ever
migrate to V1: `pip install transferqueue` into a **cloned** env first and
re-run `check_env.py`.

### 8.3 `rollout.max_model_len` unset → vLLM sized for a 262K context

`vllm_async_server.py:954` falls back to the model's
`max_position_embeddings` = 262,144 when `rollout.max_model_len` is null. One
262K sequence needs ~36 G of KV cache (36 layers × 8 KV heads × 128 head_dim
× K+V × bf16 ≈ 144 KB/token) — more than the entire 0.45-util vLLM budget, so
the engine refuses to init (and anything short of that still plans block
tables for contexts we never use). `run_grpo.sh` now always passes
`rollout.max_model_len = MAX_PROMPT_LEN + MAX_RESP_LEN`.

### 8.4 verl bug: multi-image tool responses render one placeholder (patched)

First rollout that executed a `crop_video` call died inside vLLM with
`AssertionError: Failed to apply prompt replacement for mm_items['image'][1]`.

Root cause is upstream, in `verl/experimental/agent_loop/tool_agent_loop.py`
(`_handle_processing_tools_state`): for a multimodal tool response it appends
a **single** `{"type": "image"}` content item even when `tool_response.image`
is a list — our tool returns 16 frames — while all 16 PILs are attached as
mm data. The chat template therefore renders one `<|image_pad|>` for 16
images; vLLM matches image[0] and asserts on image[1].

**Fix**: the installed file is patched in place to emit one `{"type":
"image"}` per returned image (marker comment `PATCHED (agentic-tvg`).
`preflight.sh` now refuses to start if the marker is gone, so a pip
reinstall cannot silently revert it. If verl internals ever need more than
this one-liner, do the §2 fallback properly: `git clone -b v0.9.0` +
`pip install --no-deps -e`, and carry the patch as a commit there.
