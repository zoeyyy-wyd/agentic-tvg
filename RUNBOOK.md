# Runbook — from clean checkout to trained models

Every command in order, with the expected result after each step. Verified
end-to-end on srv1-lg2 (2026-08-20/21). Companion docs: `ENVIRONMENT.md`
(env assumptions), `DATA.md` (what the data is), `README.md` (layout).

Assumed already in place (NOT covered here — see ENVIRONMENT.md):
conda env `verl` with torchcodec + the `LD_PRELOAD` fix (§7), free GPU.

---

## Step 0 — every new terminal starts with this

```bash
conda activate verl
cd ~/agentic-tvg
```

Forgetting this is the #1 error source (`ModuleNotFoundError`, wrong python).

## Step 1 — install the package, run the tests  [~1 min, no GPU/data needed]

```bash
pip install --no-deps -e .
python -m pytest tests/ -q
```

- `-e`: editable — site-packages gets a link back to this repo, so later code
  edits take effect without reinstalling.
- `--no-deps`: pip must never re-resolve torch/vllm (version lock, ENVIRONMENT.md §1).

CHECK: `24 passed`. Nothing later is worth running until this is green.

## Step 2 — three downloads, in parallel  [videos are slowest: ~20-40 min]

```bash
mkdir -p logs
nohup hf download longvideotool/LongVT-Parquet --repo-type dataset \
    --local-dir data/annotations > logs/dl_anno.log 2>&1 &
nohup hf download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir models/Qwen3-VL-4B-Instruct > logs/dl_model.log 2>&1 &
nohup bash download_videos.sh > logs/dl_videos.log 2>&1 &
```

Progress (repeat until all three are done):

```bash
tail -2 logs/dl_anno.log logs/dl_model.log logs/dl_videos.log
du -sh data/annotations models data/videos 2>/dev/null
```

CHECK: annotations ~172M (11 parquet files) · models ~8.3G (safetensors shards
+ preprocessor_config.json) · videos: rl_val ~376M / selfqa ~5.6G / tvg ~13G.

## Step 3 — build RL train/val parquet  [~2 min; needs annotations + videos]

```bash
python data_prep/extract_rl.py
```

CHECK: `kept 1668/1668` (rl_train) and `kept 114/114` (rl_val), all drop
counters zero. Output: `data/processed/rl_train.parquet`, `rl_val.parquet`.

## Step 4 — re-render SFT traces  [~3 min]

```bash
python data_prep/render_traces.py
```

CHECK: `kept 6181/6395 (train 6100 / val 81)`; the only drop reason is
`gt_out_of_range: 214` (annotations pointing past video end — expected).
Output: `data/processed/sft_train.parquet`, `sft_val.parquet`.

## Step 5 — probe demo  [~5 min; needs the model; two terminals]

Terminal A (inference server, keep it running):

```bash
bash serve_qwen3vl.sh
```

Wait for `Uvicorn running on http://127.0.0.1:8000` (~1-2 min model load).

Terminal B (Step 0 first!):

```bash
python probe/step0_probe.py --data data/processed/rl_val.parquet \
    --modes direct,tool_optional --limit 6 --out outputs/step0_demo
```

CHECK: summary table shows `format_rate 1.0` in both modes and
`tool_call_rate 1.0` in tool_optional.

**Shut the server down before any training** — Ctrl-C in terminal A, then:

```bash
nvidia-smi    # memory MUST read 0 MiB; if not: pkill -f "vllm serve", re-check
```

vLLM leaves child processes behind; killing only the wrapper is not enough.

## Step 6 — SFT smoke test: 2 real training steps  [~7 min]

```bash
EXP_NAME=sft_smoke bash run_sft_tvg.sh trainer.total_training_steps=2 \
    trainer.save_freq=-1 trainer.test_freq=-1
```

CHECK: log shows `step:1 ... train/loss:` ~2.2 then `step:2 ...` ~2.0,
`max_memory_reserved_gb` < 70. The `Kwargs passed to processor` warning spam
is harmless — filter with `grep -v "Kwargs passed"`.

Clean up the smoke checkpoint (19G):

```bash
rm -rf ckpts/sft_smoke
```

---

# Real experiments (beyond what has been smoke-tested)

## Step-0 probe, full  [~1-2 h; server up as in Step 5]

```bash
python probe/step0_probe.py --data data/processed/rl_val.parquet \
    --modes direct,tool_optional,tool_forced --out outputs/step0
```

Decision rule (plan §3): tool-call rate > 50% with clean format -> light SFT
(2K traces); chaotic calls / broken format -> full SFT (6.4K).

## SFT, for real  [~22 h full dose; server must be down]

```bash
bash run_sft_tvg.sh                                        # full: 6.1K traces
SFT_DOSE=2000 EXP_NAME=sft_tvg_light bash run_sft_tvg.sh   # light arm
```

## GRPO  [never executed yet — expect debugging]

```bash
bash run_grpo_tvg.sh
# ablations:
#   REWARD_FN=compute_score_penalty EXP_NAME=grpo_tvg_penalty bash run_grpo_tvg.sh
#   MAX_USER_TURNS=1 EXP_NAME=grpo_tvg_t1 bash run_grpo_tvg.sh
```

This is the project's largest unverified surface (plan §9 risk: LoRA + vLLM +
video + tools in one config). Budget a debugging day; run a 2-step smoke
(`trainer.total_training_steps=2`) before any long run.

## Optional — Charades-STA eval prep (pending video download)

```bash
# official Charades annotations (for the contamination check, DATA.md §8.3)
curl -sL -o /tmp/Charades.zip https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip
unzip -q /tmp/Charades.zip -d data/annotations/charades_official
# CHECK: tvg's 1859 Charades ids intersect Charades_v1_test.csv ids = 0

# then: fetch charades_sta_test.txt + Charades_v1_480.zip (AllenAI), and run
python data_prep/prepare_charades.py --sta-file <txt> --video-root <videos> \
    --out data/processed/charades_test.parquet
```

---

# Known pitfalls

1. **Forgot `conda activate verl`** -> ModuleNotFoundError / wrong python.
2. **Server not fully dead before training** -> CUDA OOM. Always verify
   `nvidia-smi` reads 0 MiB (see Step 5).
3. **Warning spam in SFT logs** (`Kwargs passed to processor.__call__`) —
   harmless, parameters do take effect; filter when reading logs.

# Disk footprint (148G volume)

env ~12G · model 8.3G · videos 18G · parquets <10M · SFT smoke ckpt 19G
(delete after) · real SFT/GRPO ckpts: budget ~30G free before long runs.
