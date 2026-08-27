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

## 6.5 CPU 内存耗尽：已解决（2026-08-27）

**结论先行**：解药是两个 glibc 环境变量，已写进 `run_grpo.sh`。降 batch 和
关 offload 都不是解法——后者是速度优化，前者只是把死亡时间往后推。

### 现象

三次崩溃，全部是 CPU RAM 打满 188G，**全部在同一位置**——一步训练结束、
唤醒 vLLM 同步权重时（`actor_rollout_update_weights` → `wake_up`）。GPU 全程
平稳且只用 39–57/80G，所以调显存旋钮无效。

| 配置 | 死在第几步 |
|---|---|
| batch 16 × K16 = 256 轨迹 | step 1（182.2G） |
| batch 8 × K16 = 128 轨迹 | step 4（182.2G） |

内存逐步累积，不是稳态高位：`111.1 → 111.8 → 129.1 → 崩`。

### 根因：glibc 的 mmap 门槛棘轮

malloc 有两条路拿内存。**堆（brk）**只能从堆顶收缩，中间的空洞被上面的块夹
着还不掉；**mmap** 是独立映射，`free` 时 `munmap` 直接还给内核，与位置无关。

走哪条由门槛决定（默认 128KB），**但这个门槛是动态的**：每释放一个 mmap
块就上调到该块大小（上限 32MB），且只升不降。本项目的分配尺寸——单帧
588KB、crop 张量 52MB——正落在这个区间。一个 52MB 的块释放一次，门槛就被
顶到 32MB 上限，此后所有中小分配改从堆里拿，free 后不归还内核。

视频帧（短命）和 rollout 缓冲（长命）交替分配，长命的把短命的空洞全锁死，
RSS 只涨不跌。

**解法**（`run_grpo.sh` 已设，`${VAR:-默认}` 形式，可被环境覆盖）：

```bash
export MALLOC_MMAP_THRESHOLD_=131072      # 钉死 128KB，关掉动态调整
export MALLOC_TRIM_THRESHOLD_=134217728   # 堆顶空闲超 128MB 归还内核
```

结尾下划线是 glibc 的命名规范，**漏了不生效且不报错**。

**实测（8 步，batch 8 × K16，其余不变）**：

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 加之前 | 111.1 | 111.8 | 129.1 | 崩 | | | | |
| 加之后 | 96.5 | 98.8 | 115.6 | 132.0 | 99.3 | 100.1 | 100.0 | 100.7 |

形态变了：从只涨不跌，变成涨到 132 后回落、并在 99–101 稳住（波动 1.4G）。
step 5 之后瞬时峰值不超过 149.8G，离 182 有 32G 余量。

### 一个判断失误（留档）

我当时看到峰值时刻 `WorkerDict 45.7G` 出现在 compute_log_prob 阶段，推断
"这是被 offload 到 CPU 的参数，关掉能省 45G"，并把它列为首要修复方向。

`param_offload=False` 的对照实验直接证伪：

| | offload=True | offload=False | 差 |
|---|---:|---:|---:|
| 内存峰值 | 175.3G | 171.3G | −4.0 |
| WorkerDict 最大 PSS | 38.4G | 37.9G | **−0.5** |
| 每步末内存 | 96.5/98.8/115.6/132.0 | 97.3/98.0/114.6/131.3 | 几乎相同 |

错在把"offload 的目的地"和"WorkerDict 的 CPU 占用"当成同一件事。那个阶段
参数本来就在 GPU 上（正在算前向），CPU 侧那 38G 是 FSDP 的内部结构和通信
缓冲，与参数放哪儿无关。

### 方法教训

3 步的 smoke 峰值 132G、看着有 56G 余量，据此判定"128 轨迹可行"——它恰好
停在悬崖前一步（step 4 就崩）。**验证长跑的 smoke 必须长于故障周期**，这里
至少 5 步。

### 测量工具

用 **PSS 而不是 RSS**。RSS 把共享页算给每一个共享者，父子进程 fork 后相加会
重复计算（"TaskRunner 75G"因此虚高）。PSS 按共享者数量均摊，求和才是真实
物理占用。采样脚本和两次实验的曲线在 `results/memtest/`。

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
两个 token_len 停在 24576 而非 32768。

**#4 的前提是错的**：它写着"rollout 阶段优化器 16G 要与 vLLM 共存，放不
下"——16G 是**全参微调**的量级。本项目是 LoRA，只有 adapter 可训练，优化器
状态实测 265 MiB，差 60 倍。常驻需求约 18G，而 vLLM 醒来后仍余 80−52=28G，
本来就放得下。offload 已于同日关闭，实测显存峰值 73.5/80G，4 步全部相同
（确定性负载，不随轨迹长短波动）。收益是每步快 91 秒。

另外 §7 整体是在"瓶颈在 GPU"的前提下写的。实测瓶颈在 CPU（见 §6.5），
GPU 全程余 20–70G，所以 #3 那类"把 GPU 余量花掉"的建议已无意义。

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
