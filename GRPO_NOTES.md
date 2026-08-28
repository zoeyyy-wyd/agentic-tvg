# GRPO Configuration Notes (verl 0.9.0 / `run_grpo.sh`)

Why each knob in `run_grpo.sh` is set the way it is. Measurements behind the
numbers: `FRAMES_SWEEP.md` and `results/memtest/`. Environment failures:
`env_setup/ENVIRONMENT.md` §8. Values quoted here are the script's current ones.

## 1. One training iteration

```
① rollout       vLLM generates TRAIN_BS × K trajectories (multi-turn agent loop)
② old log_prob  FSDP actor, forward only — the "old policy" per-token log_prob
③ advantage     group by prompt, A = (r − group mean) / group std   ← all of GRPO
④ actor update  PPO clipped objective: ratio = exp(new − old), loss = −min(rA, clip(r)A)
```

verl's trainer *is* the PPO trainer; `algorithm.adv_estimator=grpo` only swaps
step ③ (and drops the critic). Step ④ is still PPO, which is why every actor-side
key is named `ppo_*`.

## 2. K — `GROUP_SIZE` / `rollout.n` = 16

K trajectories are sampled per prompt and form one group; the advantage is each
trajectory's score relative to its group, so K ≥ 2 is required. If a group is
uniformly right or uniformly wrong its variance is zero and it contributes no
gradient — measured at 0.005% of the pool for this policy (DATA.md §6), so no
difficulty filtering is worth a pass.

K is **free on the GPU** (vLLM queues the requests; training packs by token). It
costs wall clock linearly, and host RAM through the trajectory count — every
in-flight trajectory holds its decoded video frames as CPU tensors (§6). K is
also the term that must not shrink, since the whole advantage estimate is a
within-group comparison; batch size is the one to cut instead.

## 3. Three levels of batching

This is the easiest place to get confused, because all three are called "batch".

| Level | Key | Unit | What it triggers |
|---|---|---|---|
| train batch | `data.train_batch_size` = 8 | prompts | one rollout — the on-policy boundary |
| mini-batch | `actor.ppo_mini_batch_size` = 8 | prompts (×K internally) | **one `optimizer.step()`** |
| micro-batch | `ppo_max_token_len_per_gpu` = 24576 | tokens, packed dynamically | one forward+backward (gradient accumulation) |

- The only constraint is `train_batch_size % ppo_mini_batch_size == 0`. Updates
  per iteration = train / mini × `ppo_epochs` (default 1).
- The mini-batch unit is *prompts*, so a group is never split across two
  mini-batches (`ray_trainer.py:1328` multiplies it by `rollout.n` internally).
- There is no "accumulate until train_batch_size, then optimise". With train 32 /
  mini 8 there would be four parameter updates inside one iteration, the last
  three already slightly off-policy with PPO clipping as the safety net. We set
  **train == mini**, so each iteration is exactly one update: maximally
  on-policy, and the simplest thing to reason about.

## 4. Every token and batch parameter, in pipeline order

### Before ① — data boundary

| Key | Value | Unit | Role |
|---|---|---|---|
| `data.train_batch_size` | 8 | prompts | prompts per step |
| `rollout.n` (K) | 16 | per prompt | group size → **128 trajectories/step**, the source of every cost |
| `data.max_prompt_length` | 4608 | tokens | prompt tensor width; overflow raises (`truncation=error`) |
| `data.max_response_length` | 16384 | tokens | response tensor width |
| `data.val_batch_size` | 2 | prompts | validation loader batch — **not** a row cap (see below) |

### ① rollout — vLLM side

| Key | Value | Role |
|---|---|---|
| `rollout.max_model_len` | 20992 | KV budget per sequence. Unset it falls back to 262144, whose single-sequence KV (~36 G) exceeds the pool and vLLM refuses to start |
| `rollout.max_num_batched_tokens` | 24576 | token budget for **one engine forward**, shared by prefill and decode |
| `rollout.max_num_seqs` | 256 (default) | concurrent sequence cap |
| `rollout.enable_chunked_prefill` | True | slices long prompts so they can mix with decode |
| `rollout.gpu_memory_utilization` | 0.65 | vLLM's total VRAM share, ~52 G |
| `+rollout.limit_images` | 112 | images per request: 3 crops × 30 frames + slack |
| `mm_processor_kwargs.max_pixels` | 150528 | size of the profiling dummy images; unset, vLLM profiles at the preprocessor default 16.7M px and eats the KV pool |
| `multi_turn.max_user_turns` | 3 | at most 3 tool calls |
| `multi_turn.max_assistant_turns` | 4 | 3 tool calls + the final answer |
| `multi_turn.max_parallel_calls` | 1 | one tool call per turn |
| `multi_turn.max_tool_response_length` | 2048 | truncation of the tool's **text** return (images are not counted) |

`max_num_batched_tokens` is the vLLM scheduler's per-step token budget: prefill
chunks and decode tokens draw on the same pool, with the running queue served
first so in-flight sequences are not starved
(`vllm/v1/core/sched/scheduler.py:408/432/629`). It has a second, VLM-only role
that is easy to miss:

```python
max_num_encoder_input_tokens = encoder_cache_size = max_num_batched_tokens
# vllm/config/scheduler.py:248-249
```

The ViT's input budget *and* its encode cache are both set to this value.
Shrinking it makes vision encodings get evicted and recomputed — a sensitivity
text-only models do not have.

**`max_user_turns=3` is over-provisioned.** Across steps 1/20/40/55 (128
trajectories each) the policy called the tool exactly once in every trajectory
but one, which called it twice at step 20. `limit_images=112` and
`MAX_RESP_LEN=16384` are budgeted for the same unused 3 calls. That is slack in
the *token* budget only — crop frames are charged per actual call, so it does not
cost RAM (§6).

### ② log_prob — forward only

| Key | Value | Role |
|---|---|---|
| `rollout.log_prob_use_dynamic_bsz` | True | enable packing |
| `rollout.log_prob_max_token_len_per_gpu` | 24576 | tokens per bin (forward only, so it can be more aggressive than training) |

### ③ advantage

No token parameters — plain numpy, measured at 0.12 s. K enters via the `uid`
grouping.

### ④ update_actor

Batching in §3, packing in §5. Two things worth knowing about where they land:

- `optimizer.step()` lives in `engine/base.py:124-127`: `train_batch()` is
  zero_grad → forward_backward over every bin → optimizer_step, **exactly once
  per mini-batch**. The mini-batch count is
  `data.shape[0] // mini_batch_size_per_gpu * ppo_epochs`
  (`engine_workers.py:286`) — for us 128 // 128 × 1 = 1.
- The lr scheduler advances only after the **last mini-batch of the step**
  (`update_lr_scheduler = batch_idx == total_num_iterations - 1`,
  `engine_workers.py:306`). So whatever the train/mini ratio, the cosine curve
  follows *training steps*, and `TOTAL_STEPS` is a clean anneal denominator.

### The three `max token` keys are three different things

| | Key | Value | Who packs | Which memory |
|---|---|---|---|---|
| training | `actor.ppo_max_token_len_per_gpu` | 24576 | verl, Karmarkar-Karp, once after rollout | FSDP forward+backward activations |
| forward | `rollout.log_prob_max_token_len_per_gpu` | 24576 | same | FSDP forward-only activations |
| inference | `rollout.max_num_batched_tokens` | 24576 | **vLLM scheduler, re-formed every step** | vLLM prefill activations + ViT encode cache |

They happen to be equal, but they are three parameters, three engines, three
pools of memory; changing one does not affect the others. The first two pack
offline (all the data is in hand, so bins can be balanced optimally); the third
schedules online, without knowing what arrives next.

### Constraint graph

```
MAX_PROMPT_LEN(4608) + MAX_RESP_LEN(16384) = 20992
   │
   ├─→ rollout.max_model_len = 20992              exactly how the script computes it
   │      └─ with chunked_prefill on, max_num_batched_tokens may be smaller
   │
   └─→ ppo / log_prob_max_token_len_per_gpu ≥ 20992
          asserted at seqlen_balancing.py:384 — set it lower and the run dies

max_num_batched_tokens(24576) ≥ max_num_seqs(256)      vLLM's own check
train_batch_size(8) % ppo_mini_batch_size(8) == 0      the only batching constraint
```

Packing counts **real tokens** (`attention_mask.sum()`), not padded width:
measured 927,618 tokens / 24576 → 38 bins, ~3.4 sequences each (mean 7,247
tok/sequence). So the `≥ 20992` constraint protects the worst case — one maximal
sequence must fit in one bin — not the common case.

**Changing the frame count cascades**: prompt ≈ 27 tok/frame + ~480 floor
(FRAMES_SWEEP §1) → raise `MAX_PROMPT_LEN` → raise `max_model_len` → re-check
both `*_max_token_len_per_gpu` against the inequality above. That chain is why
the table exists.

### Open item: `val_batch_size=2` wastes 3.8×

`_validate` pads the val batch up to `rollout.agent.num_workers` = 8
(`ray_trainer.py:637-638`), and `pad_dataproto_to_divisor` pads by **duplicating
real rows** (`protocol.py:74`). The duplicates decode their videos and run
through vLLM in full; only after generation does `unpad_dataproto` discard them.

| `val_batch_size` | batches | actually generated | wasted | peak concurrency |
|---|---:|---:|---:|---:|
| 2 (current) | 57 | 456 | **342 (75%)** | 8 |
| 8 | 15 | 120 | 6 | 8 |
| unset (=114) | 1 | 120 | 6 | **114 → OOM** |

2 was chosen to hold concurrency down (validation at `test_freq=20` runs with the
training state resident). But **the pad to 8 is a floor** — 2 cannot go below it.
Peak concurrency is identical to 8; the only difference is 3.8× duplicated
trajectories. Validation fires 8 times over the run, so setting it to 8 is free.
Not changed yet: `grpo_vanilla` is running, and Hydra reads the config at launch.

## 5. Gradient accumulation and dynamic packing

`loss.backward()` only **accumulates** ∂loss/∂θ into `.grad`; parameters do not
move until `optimizer.step()`. Gradients add, so:

```
mini-batch gradient g = g₁ + … + g₃₈
per bin:  forward → backward (.grad += gᵢ) → that bin's activations freed
after 38: optimizer.step()   ← bit-for-bit the same as one giant batch
```

Only one bin's activations (≤24,576 tokens) are ever resident. That is the whole
mechanism by which VRAM is bounded — and the reason the frame sweep found VRAM
flat: more frames means longer sequences means *more bins*, while the per-bin
token ceiling, and therefore the activation peak, does not move.

Packing (`use_dynamic_bsz`) is computed once per mini-batch after rollout
finishes (`rearrange_micro_batches`, Karmarkar-Karp balancing), so the bin count
differs every step as trajectory lengths change.
`ppo_max_token_len_per_gpu` is a pure hardware knob: larger → fewer bins →
faster, more activation memory, identical math. Hard floor: the longest single
sequence, i.e. `MAX_PROMPT_LEN + MAX_RESP_LEN`. The static alternative
(`ppo_micro_batch_size_per_gpu`, a fixed sequence count per bin) is mutually
exclusive with it and a poor fit here, where RL sequence lengths vary by tens of
times.

## 6. Host RAM is the bottleneck — four OOMs, three mechanisms (2026-08-27/28)

The short version: (a) glibc's mmap-threshold ratchet made the heap grow
monotonically — fixed with two env vars; (b) offload was never the RAM cost it
looked like — disabled anyway, for speed; (c) the residual slow climb is Ray's
plasma arena in /dev/shm, whose tmpfs pages are never returned once touched —
capped with `object_store_memory=40G`. Details in order below, wrong turns
included.

`rollout.agent.num_workers` (default 8) are the CPU-side rollout processes:
decode the global video → build the prompt → call vLLM over HTTP → parse the
tool call → run `crop_video` → loop. Each runs asyncio concurrency over a ≤32
thread decode pool, so ~256 concurrent video decodes by default. (Unrelated to
the PyTorch DataLoader's `num_workers`.)

**GPU was never the constraint**: 45–57 G of 80 G throughout. Three consecutive
crashes were all CPU RAM hitting 188 G, and all at the same point — end of a
training step, waking vLLM to sync weights (`actor_rollout_update_weights` →
`wake_up`).

| Config | Died at |
|---|---|
| batch 16 × K16 = 256 trajectories | step 1 (182.2 G) |
| batch 8 × K16 = 128 trajectories | step 4 (182.2 G) |

Memory ratcheted rather than sitting high: `111.1 → 111.8 → 129.1 → dead`.

### Root cause: glibc's mmap threshold ratchets

malloc has two sources. The **heap (brk)** can only shrink from the top, so a
freed block with live blocks above it is not returned. **mmap** allocations are
independent mappings that `munmap` returns to the kernel on `free`, regardless of
position.

Which one is used depends on a threshold (default 128 KB) — and that threshold is
**dynamic**: every time an mmap'd block is freed it is raised to that block's
size, up to 32 MB, and it never comes back down. This project's allocation sizes
sit right in that range: 588 KB per frame, 52 MB per crop tensor. One 52 MB free
pushes the threshold to its 32 MB ceiling, and from then on everything smaller
comes from the heap and is never returned. Short-lived video frames interleaved
with long-lived rollout buffers means the long-lived ones pin the gaps, and RSS
only climbs.

**Fix** (set in `run_grpo.sh`, in `${VAR:-default}` form so it stays overridable):

```bash
export MALLOC_MMAP_THRESHOLD_=131072      # pin at 128 KB, disable the ratchet
export MALLOC_TRIM_THRESHOLD_=134217728   # return heap top to the kernel past 128 MB free
```

The trailing underscore is glibc's naming convention. **Omit it and the variable
is silently ignored.**

Measured over 8 steps (batch 8 × K16, nothing else changed):

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| before | 111.1 | 111.8 | 129.1 | dead | | | | |
| after | 96.5 | 98.8 | 115.6 | 132.0 | 99.3 | 100.1 | 100.0 | 100.7 |

The shape changed: instead of climbing monotonically it peaks at 132 and falls
back, settling at 99–101 G (±1.4 G). After step 5 the instantaneous peak stays
under 149.8 G, 32 G clear of the wall.

### One wrong hypothesis, kept on the record

Seeing `WorkerDict 45.7 G` at the peak during `compute_log_prob`, I concluded
those were parameters offloaded to CPU and that disabling offload would recover
45 G. A controlled run disproved it outright:

| | offload=True | offload=False | Δ |
|---|---:|---:|---:|
| peak memory | 175.3 G | 171.3 G | −4.0 |
| WorkerDict peak PSS | 38.4 G | 37.9 G | **−0.5** |
| per-step end | 96.5/98.8/115.6/132.0 | 97.3/98.0/114.6/131.3 | ~identical |

The error was equating "where offload puts things" with "what WorkerDict holds on
the CPU". At that phase the parameters are on the GPU by definition — the forward
pass is running. The 38 G is FSDP scaffolding and communication buffers, which
exist wherever the parameters live. (Offload was disabled anyway, for speed: §7.)

### The fifth OOM and the plasma cap (2026-08-28)

The glibc fix was not the whole story. With it in place the 267-step run
reached **step 66** (up from 4) and then died the same way — 183 G, this time
during *rollout*, not weight sync. The TB memory series showed why the crash
point wanders: the per-step baseline still crept `97-100 → 103-104` G over 60
steps (~2 G / 20 steps), so *any* routine peak eventually crosses the line.

The carrier this time was `/dev/shm`. Three numbers that are routinely
conflated, and were during this hunt:

- `df /dev/shm` — the tmpfs, **including** segments unlinked but still mapped
  (`torch_… (deleted)`) — 17-20 G;
- `du /dev/shm` — only what still has a name — 5 MB, wildly misleading;
- `Shmem` in /proc/meminfo — df's number **plus** non-tmpfs shared maps
  (CUDA IPC) — 28-31 G. The early "shm grew 6 G in 8 steps" readings were
  this metric, i.e. partially something else.

`ray memory --stats-only` on the live run settled it: **one object, 16.5 G**
— a whole step's DataProto (128 trajectories with their pixel tensors) is a
single plasma object, alive while it crosses TaskRunner → AgentLoopWorker →
RewardLoop → actor, then freed. But plasma's arena is a tmpfs file that only
ever grows to its quota (default 30% of RAM = 53.4 G here; every ray worker
maps the same arena, so per-process numbers lie), and touched pages are never
returned. The ceiling is the cost, so cap the ceiling:

```bash
+ray_kwargs.ray_init.object_store_memory=42949672960   # 40 G, in run_grpo.sh
```

40 G = two generations of the 16.5 G object in flight plus margin; /dev/shm
high water over 67 steps was 20 G, so 2x. If it is ever too small the failure
is an explicit `ObjectStoreFullError`, not a silent OOM. (`ray_init` is
**kwargs-forwarded to `ray.init()` — main_ppo.py:75 — and the key is real per
`inspect.signature`. Beware: ray.init itself takes **kwargs, so a typo here is
swallowed silently; verify with `df /dev/shm` after start.)

### Method lessons

- **A smoke must outlive the failure period it is meant to rule out.** The 3-step
  smoke peaked at 132 G and looked like it had 56 G of headroom. It had stopped
  one step short of the cliff.
- **Measure PSS, not RSS.** RSS charges a shared page to every process mapping
  it, so summing over a parent and its forked children double-counts — the
  "TaskRunner is using 75 G" reading that started this investigation was an RSS
  artifact. PSS divides each page by its sharers, so the sum is real physical
  usage. Sampler and both runs' curves: `results/memtest/`.

## 7. What was done with the GPU headroom

The sweep left ~40 G of the card unused, and §7 originally listed four ways to
spend it. Once the bottleneck turned out to be CPU RAM (§6), most of them stopped
mattering. Current state:

- **`GPU_MEM_UTIL` 0.45 → 0.65** — done. KV pool 23 G → 39 G, roughly doubling
  rollout concurrency, which is the largest share of step time. Safe because the
  phases are disjoint: the actor is out of the way during rollout, vLLM sleeps
  during training.
- **`enable_prefix_caching`** — verl enables it by default, nothing to do. GRPO
  benefits structurally: the K trajectories of a group share one video prompt
  prefix, so its KV is computed and stored once.
- **`ppo`/`log_prob_max_token_len_per_gpu`** — left at 24576, not raised to
  32768. Fewer bins would speed up training, but the constraint is not GPU time.
- **`param_offload` / `optimizer_offload` → False** — done, and the reasoning
  that had kept them on was wrong. It assumed the optimizer needed ~16 G resident
  alongside vLLM; that is the full-finetuning figure. This is LoRA — only the
  adapter is trainable, and the optimizer state measures 265 MiB. Resident need
  is ~18 G against the 80 − 52 = 28 G that survives vLLM waking. Measured after
  the change: 73.5 / 80 G peak, identical on all four measured steps
  (deterministic, not load-dependent), and 91 s/step faster from not shuttling
  35 G of parameters across PCIe twice per step. If that margin is ever too thin,
  `gpu_memory_utilization=0.55` buys back ~8 G.

## 8. Miscellaneous

- **Do not export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` here.**
  (`run_sft.sh` does, because SFT has no vLLM.) verl toggles it at runtime inside
  the hybrid worker: on during training phases for fragmentation control, off
  around vLLM wake and weight sync, where it conflicts with sleep-mode
  CuMemAllocator. A global export leaks into the vLLM process and hits exactly
  that conflict. `engine_workers.py:760/805`.
- **`agentic_tvg` is an editable install** — a `.py` edit takes effect on the next
  process start, no reinstall (only `pyproject` changes need one). But the strings
  in `prompts.py` / `constants.py` are already baked into the parquet files:
  changing them requires re-rendering the data per DATA.md §0.5. Code taking effect is
  not the same as data being consistent.
- **verl itself is a wheel install**, and the multi-image tool-response fix is a
  direct edit to site-packages (ENVIRONMENT.md §8.4). A pip reinstall wipes it;
  preflight guards against running without it.
- **Judge credits are an operational dependency.** The run hard-stops with
  `JudgeUnavailable` when the Anthropic API refuses — by design (a silent
  fallback to the alias matcher would swap scoring instruments mid-run). It
  happened for real on 2026-08-28: ~$0.3/step at 128 trajectories, and the
  key draws from **console.anthropic.com** credit balance — claude.ai
  "usage credits" are a different pool with the same error text. Budget ~$7-10
  per 200 steps; the cache (`judge_cache.jsonl`) makes replays free.
- **A grad_norm spike is not always trouble.** Step 66: 0.131 vs 0.055
  baseline, self-corrected next step. Cause: several groups went
  near-unanimous, and group normalisation drives a lone dissenter's advantage
  toward its cap (1-vs-15 split ⇒ |A| = √15 ≈ 3.87; observed −3.4), so one
  trajectory carries its whole group's gradient. Watch for the *pattern*
  (sustained rise + falling entropy = collapse loop), not the event.
- **Config keys that are wrong but legal fail silently.** The PPO trainer reads
  `trainer.max_actor_ckpt_to_keep`; the SFT trainer's name is
  `trainer.max_ckpt_to_keep`. Copying the SFT name over cost a run on 2026-08-27
  — Hydra's `+` created the key, nothing read it, and 17 G checkpoints piled up
  until the disk hit 93%. Verify with `ls results/<run>/ckpt` after the second
  save, not by re-reading the override list.
