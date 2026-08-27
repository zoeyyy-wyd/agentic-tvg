# GRPO 训练配置原理笔记（verl 0.9.0 / run_grpo.sh）

2026-08-26 帧数压测期间整理。测量数据见 `FRAMES_SWEEP.md`，
环境问题见 `env_setup/ENVIRONMENT.md` §8。

## 1. 一个训练迭代的完整时序

```
① rollout      vLLM 生成 TRAIN_BS × K 条轨迹（多轮工具调用，agent loop）
② old log_prob FSDP actor 纯前向，记录"旧策略"逐 token log_prob
③ 算奖励+优势  按 prompt 分组，A = (r - 组均值) / 组标准差   ← GRPO 的全部
④ actor 更新   PPO 裁剪目标：ratio=exp(new-old)，loss=-min(rA, clip(r)A)
```

verl 的 trainer 本体是 PPO trainer，GRPO 只是 `algorithm.adv_estimator=grpo`
——换掉③的优势算法（去掉 critic），④仍是 PPO 更新，所以 actor 侧配置都叫
`ppo_*`。

## 2. K（GROUP_SIZE / rollout.n）

- 同一 prompt 并行采 K 条轨迹 = 一个 group；优势用组内相对分数算，K≥2。
- 组内全对/全错 → 方差 0 → 无梯度信号，样本被跳过（judge 时代实测
  仅 0.005%，见 DATA.md §6）。K 越大，难题越可能出现组内至少一条对 → 有效信号。
- 成本：**GPU 显存基本免费**（vLLM 排队、训练按 token 装箱）；代价是每步
  墙钟时间 ∝ K，以及 RAM ∝ 轨迹数×帧数（每条轨迹独立解码/持有视频张量）。

## 3. 三层批次结构（最易混淆处）

| 层级 | 配置 | 单位 | 触发的动作 |
|---|---|---|---|
| train batch | `data.train_batch_size` | prompt | ① rollout 采数据（on-policy 边界） |
| mini-batch | `actor.ppo_mini_batch_size` | prompt（内部自动 ×K） | **一次 optimizer.step** |
| micro-batch | `ppo_max_token_len_per_gpu` 动态装箱 | token | 一次 forward+backward（梯度累积） |

- 约束只有一个：`train_batch_size % ppo_mini_batch_size == 0`。
  每迭代更新次数 = train/mini（×`ppo_epochs`，默认 1）。
- mini 单位是 prompt → 一个组永远不会被拆进两个 mini-batch
  （ray_trainer.py:1328 `ppo_mini_batch_size *= rollout.n`）。
- **不存在"攒够 train_batch_size 才 optimize"**：train 32 / mini 8 时，
  一次迭代内部就有 4 次参数更新，后 3 次已轻微 off-policy，靠 PPO clip
  兜底。我们配置 train==mini（1:1）→ 每迭代恰 1 次更新，最保守 on-policy。

## 4. backward ≠ 参数更新（梯度累积）

`loss.backward()` 只把 ∂loss/∂θ **累加**进 `.grad`，参数不动；
`optimizer.step()` 才改参数。梯度可加，所以：

```
mini-batch 梯度 g = g₁+...+g₁₀
每箱: forward → backward(.grad += gᵢ) → 该箱激活立即释放
10 箱后: optimizer.step()  ← 与一次算完整个 mini-batch 逐位等价
```

任意时刻卡上只有一箱（≤16384 token）的激活——显存被控制的全部原理。

## 5. 动态装箱（use_dynamic_bsz）

- 时机：**rollout 全部结束后**，每个 mini-batch 更新前一次性算好
  （`rearrange_micro_batches`，Karmarkar-Karp 均衡分箱）。
- 箱数 = `ceil(mini-batch 内 token 总数 / ppo_max_token_len_per_gpu)`，
  每步都不同（轨迹长短随模型行为变）。smoke 实测：150K token / 16384
  → 10 箱。
- **这就是帧数扫描显存平坦的原因**：帧数↑ → 序列长 → 箱数多，
  单箱 token 上限不变 → 单次前向激活峰值不变。
- `ppo_max_token_len_per_gpu` 是纯硬件旋钮：调大→箱少→更快→激活显存高，
  数学结果不变。硬约束：≥ 最长单条序列（MAX_PROMPT_LEN+MAX_RESP_LEN）。
- `log_prob_max_token_len_per_gpu`：同逻辑用于②（纯前向，可更激进）。
- 静态模式 `ppo_micro_batch_size_per_gpu`（固定每箱条数）与动态二选一；
  RL 序列长短差几十倍，动态是唯一合理选择。

## 6. AgentLoopWorker（rollout.agent.num_workers，默认 8）

rollout 的 CPU 侧执行进程：解码全局视频 → 拼 prompt → HTTP 调 vLLM →
解析 tool call → 执行 crop_video → 拼回对话循环。每进程 asyncio 并发 +
≤32 线程解码池 → 默认最多 ~256 路并发视频解码。**高帧数下 RAM 的第一
杀手**（第二杀手是 rollout 产物 pixel tensor ∝ TRAIN_BS×K×F，限流救不了，
见 FRAMES_SWEEP §3）。与 PyTorch DataLoader 的 num_workers 无关。

**2026-08-27 实测证实了"第二杀手"，而且它才是真正的上限**：F=128 下
TRAIN_BS=16×K=16（256 轨迹）在 step 1 的 vLLM 权重同步处 OOM——188G 的机器
用到 181G，其中 TaskRunner 独占 95.8G。降到 8×16（128 轨迹）峰值 132G，
留 56G 余量。**瓶颈完全在 CPU，GPU 全程只用 45–57G/80G**，所以调
`GPU_MEM_UTIL` 之类的旋钮无效，唯一有效的是每步轨迹数。

## 6.5 CPU 内存耗尽：未结案（2026-08-27）

**现象**：三次崩溃，全部是 CPU RAM 打满 188G，**全部发生在同一位置** ——
一步训练结束、唤醒 vLLM 同步权重时（`actor_rollout_update_weights` →
`wake_up`）。GPU 全程平稳且只用 39/80G，所以调显存旋钮无效。

| 配置 | 死在第几步 |
|---|---|
| batch 16 × K16 = 256 轨迹 | step 1（182.2G） |
| batch 8 × K16 = 128 轨迹 | step 4（182.2G） |

**已确认的事实**

1. 逐步累积，不是稳态高位：111.1 → 111.8 → 129.1 → 崩。
2. 轨迹数减半，内存只降 21%（TaskRunner 95.8G → 75.3G）。**大部分占用与
   批次无关**，所以降 batch 治标不治本。
3. 峰值构成（PSS，已按共享者均摊，无重复计算）：actor ~97G ·
   fork 出的 DataLoader worker ~17G · `/dev/shm` 30.6G。
4. `ps` 的 RSS 会把父子共享页重复计算（TaskRunner 那 75G 虚高），但 PSS
   证明 fork 出来的 worker 确实持有 17G 真内存。
5. Ray 对象存储和 rollout dump 都已排除（前者 3MB，后者每步 0.7MB）。

**方法教训**：3 步的 smoke 峰值 132G、看着有 56G 余量，据此判定"128 轨迹
可行"是错的——它恰好停在悬崖前一步。**验证长跑的 smoke 必须长于故障周期**，
这里至少要 5 步。

**已验证有效的缓解**：设 glibc 调优变量，step-1 末内存 111.05 → 96.49G
（−13%）：

```bash
export MALLOC_MMAP_THRESHOLD_=131072      # 钉死阈值，禁用"棘轮"
export MALLOC_TRIM_THRESHOLD_=134217728   # 堆顶空闲超 128MB 归还内核
```

原理：glibc 的 mmap 阈值是动态的，每释放一个大块就上调（上限 32MB）且
**只升不降**。本项目的分配尺寸（单帧 588KB、crop 52MB）正落在这个区间，
一旦阈值抬到 32MB，后续分配改从堆里拿，free 后不归还内核，RSS 只涨不跌。

**未结案**：只跑到 step 1 就没机器了。降了 13% 但采样峰值仍到 127.4G，
**不知道逐步增长是否消失**——上次死在 step 4，必须跑到 5 步以上才有结论。

**判断（2026-08-27，未验证）：offload 的方向是反的。**

06:48 采到的峰值把话说死了——最吃 CPU 的那一刻，GPU 几乎空着：

```
总 147.0 GB / 188        WorkerDict 52.4G（compute_log_prob 阶段）
                         TaskRunner 36.9G · /dev/shm 30.6G
同一时刻 GPU 11.4 / 80 GB
```

| | 实测用量 | 余量 |
|---|---|---|
| GPU | 39–57 / 80 GB | 一直空着 20–70 GB |
| CPU | 147 / 188 GB 且在涨 | 三次崩溃 |

`param_offload` + `optimizer_offload` 的前提是"GPU 紧、CPU 松"，而这里恰好
相反：我们在拿稀缺的 CPU 内存去省一个不缺的 GPU 显存。

所以这不是单一泄漏，是结构性的——一个进程树里同时住着 FSDP 训练态、vLLM
引擎、128 条多模态 rollout 缓冲，还用 CPU RAM 当 GPU 的交换空间。三项叠加
才越线：基线高（稳态 ~130G）+ 权重同步的瞬时尖峰 + 逐步漂移。glibc 只解决
了漂移中的约 13%，是三项里最小的一项。

**下次优先试这个，而不是继续调 glibc 或降 batch**：

```bash
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
```

一次去掉两样：52G 常驻，以及每步搬运 16G fp32 参数产生的 pinned memory
churn（最可疑的漂移源）。

放得下吗：vLLM 占 0.65×80 = 52G，actor 训练态约 24G，合计 76 < 80——紧但
可能刚好。装不下就把 `gpu_memory_utilization` 降到 0.5，rollout 慢一点换
CPU 活命。§7 那份"GPU 余量怎么花"的建议表是在"瓶颈在 GPU"的前提下写的，
已经不适用。

置信度：「瓶颈在 CPU 不在 GPU」确定，有测量；「offload 是主要贡献者」较有
把握（WorkerDict 52.4G 是实测的单个最大项）；「关掉就能跑通」未验证。

**下次从这里继续**：

```bash
export MALLOC_MMAP_THRESHOLD_=131072 MALLOC_TRIM_THRESHOLD_=134217728
EXP_NAME=memtest bash run_grpo.sh trainer.total_training_steps=8 \
    trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1
```

看每步的 `actor/perf/cpu_memory_used_gb` 斜率：

- 平了 → glibc 是主因，把两个变量写进 `run_grpo.sh` 即可开正式跑
- 仍涨、`/dev/shm` 同步涨 → 查 shm 段泄漏（`/proc/*/fd` 里的
  `torch_*(deleted)`）
- 仍涨、shm 平 → 最可能是 pinned memory 池：`param_offload=True` +
  `optimizer_offload=True` 每步搬运 16G fp32 参数，PyTorch 的
  `CachingHostAllocator` 在进程生命周期内**从不归还**页锁定内存。GPU 尚有
  40G 空闲，可试着关掉 offload。

## 7. GPU 余量（80G 只用 ~39G）怎么花

按性价比：

1. `GPU_MEM_UTIL` 0.45→0.65：KV 池 23G→39G，rollout 并发翻倍（步时大头）。
   安全：rollout 时 actor 已 offload，训练时 vLLM 睡眠，两相错开。
2. `enable_prefix_caching`：**verl 默认已开**，无需动。GRPO 天然受益——
   组内 K 条轨迹共享同一视频 prompt 前缀，KV 只算/存一份。
3. `ppo/log_prob_max_token_len_per_gpu` 16384→32768：箱数减半，训练/重算
   提速；激活 ~30G→~50G，训练阶段独占 GPU 放得下。
4. 关 `optimizer_offload`（省 CPU↔GPU 搬运）与 #1 冲突：rollout 阶段
   优化器 16G 要与 vLLM 共存，0.65 util 下放不下。优先 #1，offload 保持。

**现状（2026-08-27）**：#1 已落地（`GPU_MEM_UTIL=0.65`），#2 本就默认开，
两个 token_len 停在 24576 而非 32768。#3/#4 现在都没有意义——见 §6，余量
在 GPU 上，瓶颈在 CPU，再往 GPU 要空间换不来任何东西。

## 8. 杂项

- 显存碎片：GRPO 侧**不要**手动 export `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`（run_sft.sh 加了是因为 SFT 无 vLLM）。verl 在
  hybrid worker 里运行时动态开关：训练阶段开（防碎片），vLLM 唤醒/权重
  同步窗口关（睡眠模式 CuMemAllocator 冲突）。engine_workers.py:760/805。

- `agentic_tvg` 是 editable 安装：改 `.py` 下次启动进程即生效，无需重装
  （改 pyproject 才需要）。但 `prompts.py`/`constants.py` 的字符串已烘焙
  进 parquet——改它们必须按 PLAN §4 重渲染数据，代码生效≠数据一致。
- verl 本体是 wheel 安装：多图补丁直接改了 site-packages
  （ENVIRONMENT.md §8.4），pip 重装会冲掉，preflight 有守卫。
