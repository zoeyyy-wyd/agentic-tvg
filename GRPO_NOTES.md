# GRPO mechanics notes (verl 0.9.0 / `run_grpo.sh`)

Three parts: the process/memory picture (§1–2), the four OOMs analysed against
it (§3), and what every config key means (§4). All numbers were measured on
this box (1×A100 80G, 188G RAM).

---

## 1. The processes: who runs, what they hold

| Process | Job | Resident even when idle |
|---|---|---|
| **TaskRunner** | orchestrator: dispatch, collect, compute advantages | a few G; heap balloons while working |
| **WorkerDict** | FSDP actor: forward / backward / optimizer | **~37G** (FSDP comm buffers and shard bookkeeping — present wherever the params live) |
| **vLLMHttpServer** | the engine's "front door", a Ray actor | CPU side ~15G total; |
| **VLLM::EngineCore / Worker** | vLLM's own subprocesses (ZMQ-linked, *not* Ray actors — a millisecond scheduling loop cannot afford Ray's per-message serialization) | GPU side 52G while awake |
| **AgentLoopWorker × 8** | rollout executors: decode video, drive vLLM over HTTP, run the crop tool | 1–2G each, ballooning while decoding |
| **RewardLoopWorker × 8** | call the judge API | ~0.6G each |
| torch DataLoader worker × 8 | forked from TaskRunner; read parquet, build text, **drop the video column** (popped in rl_dataset.py) — nearly idle in GRPO | ~17G PSS total (mostly their pro-rated share of pages inherited from the parent, not own data) |
| OS / raylet / caches | | ~10G |

**Fixed base: ~80–90G. Ray's watchdog kills at 188 × 95% = 179G.**

One more region: the **plasma conveyor in `/dev/shm`** — one big file Ray
opens; processes map it to pass bulk data (§2, §3b).

Two easily-confused names: `data.dataloader_num_workers` (the near-idle row
above) and `rollout.agent.num_workers` (the AgentLoopWorkers, where the heavy
lifting is) are **unrelated**.

**None of this architecture exists without tool calling**: in sync mode
(verl's default) the vLLM engine lives *inside* the WorkerDict process and
generation is a function call — no HTTP, no AgentLoopWorker. The single
requirement "generation must pause mid-stream → really execute a tool → feed
the result back and continue" is what forces vLLM into a standalone service.

## 2. One training step (~785s): flow and memory actions

### ① rollout (gen, ~150s)

```
TaskRunner reads 8 parquet rows (pure text, a few KB)
  → repeat(n=16, interleave): [p1×16, p2×16, …] = 128 rows
  → chunk(8): contiguous split → each AgentLoopWorker happens to get
    one prompt's whole group of 16 (a numerical coincidence of
    batch = num_workers; side effect: vLLM's prefix cache shares the
    video-prefix KV across the group)

Inside an AgentLoopWorker (16 conversations each):
  A conversation is just data (message list + state); it does not run.
  ONE asyncio event-loop thread picks the 16 up in turn — they spend
  most of their life waiting for vLLM's HTTP reply, and waiting costs
  no thread. The loop thread only steps in for milliseconds when a
  wait completes (parse a tool_call, fire the next request).
  Decoding is seconds of pure CPU — it would freeze all 16 if done on
  the loop thread → thrown into a ≤32-thread decode pool (resident
  workers + a task queue; torchcodec is C code that releases the GIL,
  so the threads truly run in parallel).
  Machine-wide ceiling: 8 workers × 32 threads ≈ 256 concurrent decodes.

One trajectory's journey:
  decode 128 global frames (~74MB) → pixels ride the HTTP body to vLLM
  → vLLM preprocesses its own copy → cudaMemcpy to GPU → stream tokens
  → <tool_call> appears → worker really decodes 30 frames (52MB) → next
  round → <answer> or the 3+1-turn cap → assemble the finished item
  → serialize onto the /dev/shm conveyor (~130MB parcel)
  → TaskRunner takes it, deserializes into its own heap, accumulates
    the ~16.5G batch
```

**Serialize** = flatten an object into one contiguous byte stream (only bytes
cross process boundaries); **deserialize** = the receiver rebuilds the object
in *its own heap* from those bytes. The flattened copy and the rebuilt copy
are different memory — so during every hand-off window the same data exists
three times: sender's heap, conveyor, receiver's heap.

### ②–⑥ training side

| Phase | Time | Memory action |
|---|---|---|
| ② old_log_prob | ~172s | TaskRunner serializes the **whole 16.5G batch** → single big object on the conveyor → WorkerDict deserializes another copy → forward in 24576-token bins |
| ③ ref (for KL) | ~176s | **a separate remote call — the batch rides the conveyor again in full** (WorkerDict keeps nothing across calls); forward with the adapter off |
| ④ advantages | 0.1s | scalar math, negligible |
| ⑤ update_actor | ~346s | conveyor again; forward+backward, LoRA grads are 33M params, optimizer state 265MB, activations bounded by checkpointing |
| ⑥ update_weights | ~15s | FSDP gathers params layer-by-layer (layered_summon) → bf16 → CUDA IPC to the waking vLLM (which re-claims its 52G of VRAM) |

The big object on the conveyor is **strictly sequential — one alive at a
time** (`ray memory` reads "1 objects, ~16.5G" every time). Each step has 3–4
hand-off windows with a triple-copy moment; stacked on vLLM's residents and
FSDP scaffolding that is the **~50G transient crest**.

**The `cpu_memory_used_gb` curve is sampled once per step, at step end** — it
photographs low tide (96–133G). The crest lands between photographs (Ray's
crash reports say 182–183G). "It died at 133" is a sampling artifact.

## 3. The four OOMs: tide and waves

```
crash condition = baseline (tide, creeps up every step)
                + transient stacking (wave, ~50G, roughly constant)  > 179G
```

Nothing abnormal happens on the step that dies — its wave is the same size as
every previous step's. **The anomaly is amortized over all earlier steps**:
each left behind a little memory that was "freed" but never returned to the
kernel. `free()` updates the program's private ledger; Ray's killer reads the
kernel's. Two grow-only mechanisms:

### 3a. glibc's heap ratchet (each process's own heap)

The heap is a notebook that can only be torn from the end: a crossed-out
entry in the middle stays under live entries above it. Large blocks normally
use mmap "sticky notes" (torn = truly returned), but glibc's size threshold is
dynamic — freeing a big note raises the threshold to that note's size (cap
32MB), **and it never comes back down**. One free of a 52MB crop tensor pins
it at 32MB → every 588KB frame tensor thereafter goes into the notebook →
crossed out but trapped → the notebook thickens a little every step.

- Symptom: step-end 111 → 112 → 129 → dead (step 4, 182G).
- Fix (set in run_grpo.sh; the trailing underscore is glibc's convention —
  omit it and the variable is **silently ignored**):

```bash
export MALLOC_MMAP_THRESHOLD_=131072      # pin the threshold, disable the ratchet
export MALLOC_TRIM_THRESHOLD_=134217728   # return heap-top slack >128MB to the kernel
```

- After: same config, 8 steps: 96→99→116→**132→99**→100→100→101 — it climbs
  and comes back down.

### 3b. the plasma arena (`/dev/shm` conveyor)

A tmpfs file consumes a physical page **the first time that page is written,
and keeps it forever after** (a hotel that builds rooms on demand and never
demolishes: checkout only edits Ray's front-desk ledger). Physical cost = the
furthest offset ever built. Parcel sizes vary per step (response lengths are
random → the DataProto fattens and slims), so a new parcel occasionally fails
to fit the vacant pattern and builds further out: 17 → 20G over 67 steps,
with the default quota (30% of RAM = 53.4G) as the only true bound.

- Fix: `+ray_kwargs.ray_init.object_store_memory=25769803776` (24 GiB).
  Basis: the big object is sequential, one at a time (16.5G), so 24G holds it
  — measured stable at 20G for 100+ steps after capping. Overflow is an
  explicit `ObjectStoreFullError`, not a vague OOM; `response_length/mean`
  leaving its 3.2K baseline (itself a hacking signal) fattens the object and
  is the early warning.
- Verification matters: `ray_init` is **kwargs-forwarded to `ray.init()`,
  which itself takes **kwargs — **a typo is swallowed silently**. Confirm from
  the live store: `ray memory --stats-only` percentage back-solves the quota
  (16.3G / 66.33% → 24.0G ✓).

### 3c. The four crashes

| # | Scene | Mechanism |
|---|---|---|
| 1 | c30 smoke: 2 train steps, died in validation | training state resident + val batch on top (`val_batch_size` unset = all 114 rows in one bite) |
| 2 | batch16×K16, step 1, 182G | 256 trajectories' baseline already at the kill line; first wave over it |
| 3 | batch8×K16, step 4, 182G | ratchet raised the tide 111→129 in three steps; step 4's wave crossed |
| 4 | glibc fixed, step 66, 183G | arena growth + slow drift moved the tide 97→104; one slightly larger wave |

**The crash step moved 1 → 4 → 66**: each mechanism fixed makes the tide rise
slower and the run last longer. With both pinned, crest ≈ 150G < 179G — steps
67 → 243+ without another OOM.

### 3d. Ruled out, and method lessons

- **Offload is not where the RAM goes** (A/B: peak 175.3 vs 171.3G, WorkerDict
  PSS 38.4 vs 37.9G). Under the old config the fp32 weights *did* park in RAM,
  but at the OOM instants the weights were on the GPU anyway (a forward was
  running). It is off for **speed**: 91 s/step saved by not shuttling 35G over
  PCIe each step.
- **A smoke must outlive the failure period**: the 3-step smoke peaked at 132G
  and looked 56G safe — it stopped one step short of the cliff (step 4).
- **Measure PSS, not RSS**: RSS charges a shared page to every mapper, so
  summing a parent and its forked children double-counts ("TaskRunner uses
  75G" was this artifact). PSS pro-rates; its sum is real. Sampler and both
  runs' curves: `results/memtest/`.
- **Three "shm" numbers are three different things**: `df /dev/shm` includes
  unlinked-but-mapped segments (the real cost); `du` sees only named files
  (wild underestimate); `Shmem` in /proc/meminfo adds non-tmpfs shared maps
  (CUDA IPC) on top.

## 4. Config keys (`run_grpo.sh`, every non-default)

**Scale and grouping**
- `train_batch_size=8` × `rollout.n=16` (K) = 128 trajectories/step. K is the
  group-comparison sample size: advantage = (r − group mean)/group std —
  **K must not shrink** (it is the reliability of the within-group baseline);
  batch only sets how many distinct prompts per step. A near-unanimous group
  drives the lone dissenter's advantage toward the cap √15 ≈ 3.87 — the
  occasional grad_norm spike (measured 0.131 vs 0.055 baseline, self-corrected
  next step) is this mechanism; only a *sustained* rise with falling entropy
  is the collapse signature.
- `ppo_mini_batch_size=8` (= batch) + `ppo_epochs=1` (default): one optimizer
  step per collected batch, purely on-policy. Mini-batches are per prompt — a
  group is never split across two.
- Binning: `ppo_max_token_len_per_gpu` / `log_prob_max_token_len_per_gpu` /
  `max_num_batched_tokens` all 24576 — dynamic_bsz packs by tokens, one
  forward per full bin; the hard bound on activation memory.

**Lengths**
- `max_prompt_length=4608` (measured text 224–254 + vision ~3.9K, ~450 spare),
  `max_response_length=16384` (measured mean 3.2K, clip rate 0.1%),
  `max_model_len` = their sum, 20992. **Must stay < the 24576 bin budget** so
  any single sequence fits one bin; otherwise a long sequence cannot be split
  and activations blow VRAM. Left unset, max_model_len falls back to 262144 —
  ~36G of KV for one sequence, vLLM refuses to start.
- `truncation=error`: an over-long prompt fails loudly instead of silently.

**Optimizer and horizon** (round-2 form, 2026-09-01; the round-1 form is
kept below it because grpo-vanilla ran under it)
- **constant lr 1e-5** and the horizon is `trainer.total_epochs` (EPOCHS=2
  default; verl derives 133 steps/epoch from the 1,068-prompt loader at
  batch 8, drop_last). Constant is hygiene, not a lever — the round-1
  plateau was pool saturation, not lr decay (GRPO_v1_RESULTS §4) — but it makes
  the horizon and any resume schedule-free, which the two-stage curriculum
  (GRPO2_PLAN §3e) relies on. `TOTAL_STEPS` survives as an optional hard cap
  for short diagnostics and as stage 2's required explicit horizon.
- Round 1 ran cosine 1e-5 → 1e-6 with `total_epochs=100` as a sentinel and a
  hardcoded `TOTAL_STEPS=267` as the anneal denominator — under cosine a
  resume with a different value makes the lr jump (the scheduler checkpoints
  its step counter; the curve is rebuilt from config). Reproduce with the
  README's round-1 line; `min_lr_ratio` is inert under constant.

**KL**
- `use_kl_loss=True, coef=0.001, low_var_kl`; `use_kl_in_reward=False`. KL in
  the loss, not the reward (the GRPO paper's placement) — and it keeps the
  reward metric a pure task score, comparable to the baselines. Reference
  policy: `lora_rank>0` triggers `ref_in_actor` — the ref **is this actor
  with the adapter switched off**; zero extra VRAM, cost is one extra full
  forward per step (~176s, ~22% of step time).

**VRAM budget**
- `gpu_memory_utilization=0.65`: vLLM takes 52G awake (8G weights + 44G KV
  pool); `free_cache_engine=True` returns it while asleep — vLLM and FSDP
  time-share the GPU within a step.
- `param_offload=False / optimizer_offload=False`: LoRA's resident need is
  ~18G (fp32 weights 17.7 + grads 0.13 + optimizer 0.25) and fits the 28G
  that survives vLLM waking; offload is a full-finetune mechanism (16G of
  optimizer state). VRAM peak 73.5/80, identical on four measured steps
  (deterministic load).
- `limit_images=112` + `mm_processor_kwargs={max_pixels:150528}`: vLLM's
  multimodal budget, with profiling dummies pinned at the real crop size —
  unset, profiling assumes the preprocessor default 16.7M px and eats the KV
  pool.

**Rollout shape**
- `mode=async` + `agent.default_agent_loop=tool_agent` + `multi_turn.*`
  (hermes format, 3+1 turn cap): these keys summon the whole §1 architecture.
  `agent.num_workers=8` (default) is the CPU-side decode parallelism — the
  first knob to turn down under RAM pressure (changes no training math).
- `data.return_raw_chat=True`: templating/tokenization moves to the
  AgentLoopWorker; the dataset hands over raw text.

**Saving and validation**
- `save_freq=20` / `test_freq=20`: a save and a full 114-row validation every
  20 steps, one-to-one. `+max_actor_ckpt_to_keep=1` — the **PPO trainer reads
  only this name**; SFT's `max_ckpt_to_keep` is legal-but-unread here (it once
  silently piled two 17G checkpoints to 93% disk). A resumed process **never
  deletes the checkpoint it resumed from** — clean that one by hand.
- `val_batch_size=2`: the validation loader's **batch size, not a row cap** —
  all 114 rows still run, two at a time; unset means all 114 in one bite,
  which is crash #1.
- `rollout_data_dir` / `validation_data_dir`: **not interchangeable** — the
  first is read only by the training loop, the second only by `_validate()`;
  a val_only run with just the first writes nothing. Both accumulate
  `<step>.jsonl` (only a rerun with the same EXP_NAME overwrites).

**Entry point**
- `MODEL_PATH` defaults to `results/sft-mix/merged` — this one path feeds
  three things: vLLM's generation weights, the FSDP actor's init, and the KL
  reference. Point it wrong and all three are wrong, with no error.
