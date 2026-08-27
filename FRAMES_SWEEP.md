# GRPO 帧数/配置压力测试报告（2026-08-26）

机器：srv1-lg2，1×A100 80GB PCIe，64 核 / 188GB RAM。
方法：每档配置跑 2 个真实 GRPO step 的 smoke（`run_grpo.sh` + 覆盖项），
TRAIN_BS=8、K=4（论文配置档除外），测量 nvidia-smi 5 秒采样峰值 +
verl `perf/` 指标。目的：确定 SFT/RL 必须共享的帧数上限（PLAN §6），
以及各档所需的联动配置。

前置：本次调试先修掉了 4 个启动阻塞（详见 `env_setup/ENVIRONMENT.md` §8）：
/etc/hosts 缺 localhost → Ray 全体超时；verl V1 trainer 缺 `transfer_queue`
依赖 → `use_v1=False`；`max_model_len` 未设时回退 262144 → 显式传
prompt+response；verl 0.9.0 多图工具响应只渲染 1 个占位符 → site-packages
补丁 + preflight 守卫。

## 1. token 成本实测（CPU，与 verl 路径逐字节一致）

全局视图（≈27.2 tok/帧，含时间戳文本；文本+工具 schema 底数 ≈480 tok）：

| GLOBAL_FRAMES | prompt tokens | 需要 MAX_PROMPT_LEN |
|---|---|---|
| 64（生产现值） | 2,203 | 4096（现值）|
| 96 | 3,074 | 4096 |
| 128 | 3,944 | 4608 |
| 160 | 4,816 | 5632 |
| 192 | 5,687 | 6144 |
| 256 | 7,428 | 8192 |
| 512 | ~14,500 | 16384 |

crop 工具返回（每帧 ~94 tok 实测 / ~147 tok 最坏比例；3 次调用计）：

| CROP_NUM_FRAMES | 单次 crop | 3 次（最坏） | 需要 MAX_RESP_LEN | 需要 limit_images |
|---|---|---|---|---|
| 16（现值） | 1,524 | ~7,500 | 8192（现值，紧） | 64 |
| 24 | 2,268 | ~11,100 | 12288 | 80 |
| 32 | 3,012 | ~14,700 | 16384 | 112 |

## 2. 扫描结果（全部 2/2 step 通过，除论文档）

| 配置 | MAX_PROMPT_LEN | GPU 峰值 | 训练侧 alloc/reserved | RAM 峰值 | step 时长 | score/mean |
|---|---|---|---|---|---|---|
| F=64 K=4 | 4096 | 39.3 GB | 29.2 / 31.3 GB | ~65 GB | 131 s | 0.32 |
| F=128 K=4 | 4608 | 38.4 GB | 30.1 / 32.3 GB | 65.6 GB | 161 s | 0.28 |
| F=192 K=4 | 6144 | 38.8 GB | 31.6 / 33.9 GB | 73.0 GB | 210 s | 0.33 |
| F=256 K=4 | 8192 | 39.0 GB | 31.0 / 33.2 GB | 76.5 GB | 242 s | 0.35 |
| F=512 K=16（LongVT 论文档） | 16384（+RESP 16384，token 上限 32768） | ~39.6 GB（rollout 中） | 未到训练阶段 | **>180 GB，系统 OOM** | — | — |
| F=512 K=16 尝试4（TRAIN_BS=4=64条, workers=2） | 同上 | 48.7 GB | （30+ min 未出 step1，主动终止） | 101 GB ✓ | **>30 min/步 ✗** | — |
| **F=128 K=16 优化组合**（util 0.65, token_len 32768, 128条/步） | 4608 | **55.0 GB** | 42.8 / 45.9 GB | 113.6 GB | **632 s** | 0.46/0.28 |
| **C=30 crop 预算验证**（F=64, K=4, RESP 16384, limit_images 112, 装箱 24576, util 0.65, mm_processor_kwargs.max_pixels=150528） | 4096 | — | 35.5 / 38.0 GB | 70.7 GB | 通过 2/2 | 0.30 |

GPU 峰值中 36 GB 是 `GPU_MEM_UTIL=0.45` 锁死的 vLLM 常驻（训练时睡眠
释放），训练侧激活由 `ppo_max_token_len_per_gpu=16384` 的 token 装箱决定
——**所以 GPU 显存对全局帧数几乎无感**，F=64→256 只涨了 2 GB 训练侧。

## 3. LongVT 论文配置（512 帧 / K=16 / 16K 生成）的三连失败

1. **数据物理限制**：RL 视频是低帧率转码（总帧数 min=447 / median=615 /
   max=903），`nframes=512` 超过部分视频总帧数 → qwen-vl-utils 报错，且
   fallback 到已被移除 `read_video` 的 torchvision，报出误导性
   AttributeError。修复：parquet 按每条视频实际帧数封顶
   （f512 有 163/893 行被封）。**推论：F≤446 时全数据集无需封顶；再高就
   必须在 extract_rl.py 做逐视频 cap。**
2. **RAM OOM（8 agent workers）**：128 条轨迹并发解码 512 帧原始分辨率
   视频，8 进程 × 32 解码线程 → RAM 180.9/188 GB，Ray 杀 9 个 worker。
3. **RAM OOM（2 agent workers）**：`rollout.agent.num_workers=2` 限流后
   仍 180.8/188 GB → 大头不是解码并发，而是 **rollout 产物本身**：
   128 条轨迹 × 512 帧 pixel tensor（float32 ≈ 280 MB/条 ≈ 36 GB）在
   batch 组装与 Ray plasma（/dev/shm，计入 RAM）中多副本流转，随
   `TRAIN_BS × K × F` 线性增长，与解码并发无关。
4. **尝试 4（半批量，主动终止）**：TRAIN_BS=4（64 条轨迹）+ num_workers=2
   下 RAM 峰值 101 GB（可控）、GPU 48.7 GB，但 30+ 分钟未完成 step 1 ——
   **论文配置在单卡上显存放得下、墙钟时间不可行**（外推 >30 min/步，150
   步 ≈ 75-100 h）。结论已足够，终止以让位后续测试。

## 3.5 优化组合验证（生产候选：F=128 + K=16）

`GPU_MEM_UTIL=0.65` + `ppo/log_prob_max_token_len_per_gpu=32768`
（prefix caching 为 verl 默认开启；expandable_segments 由 verl 运行时自管，
不要手动 export，见 run_grpo.sh 注释）。TRAIN_BS=8 × K=16 = 128 轨迹/步，
2/2 step 通过。

step 2 稳态拆解（632 s，global_seqlen 78 万 token）：
gen 201 s（32%）· old_log_prob 138 s（22%）· update_actor 278 s（44%）·
权重同步 14 s。按 4B FLOPs 与 MFU 0.34 核算，update/log_prob 已贴算力墙，
配置层无大头可省；每 token 吞吐较未优化配置略优，步时增长纯粹来自
K=16 的 4.3× token 量。

**生产时间账（150 步）**：

| 组合 | 轨迹/步 | 步时外推 | 总时长 |
|---|---|---|---|
| F=64, K=8, BS=32（原计划） | 256 | ~18 min | ~45 h |
| **F=128, K=16, BS=16（建议）** | 256 | ~21 min | ~52 h |
| F=128, K=16, BS=32 | 512 | ~42 min | ~105 h（不可行） |

单卡 GRPO 的硬约束是墙钟时间，不是显存（GPU 峰值 55/80 GB，RAM
114/188 GB）。K=16 若采纳，TRAIN_BS 应降为 16。

## 4. 结论与建议

- **80 GB 显卡不是瓶颈**。到 F=256 全程 GPU ≤39 GB；就算把单序列上限提
  到 32768（论文 token 预算），vLLM 初始化也通过了。GPU 还有 ~40 GB 余
  量，可用于将来提高 `GPU_MEM_UTIL`（加大 KV 池、提升 rollout 并发）或
  提高 `ppo_max_token_len_per_gpu`（减少梯度累积次数）。
- **真正的两个约束**：
  1. **系统 RAM（188 GB）**：随 F 和每步轨迹数（TRAIN_BS×K）线性增长。
     F≤256 + 256 条轨迹/步的生产设置安全（外推 ~90-110 GB）；F=512 +
     128 条轨迹已爆。旋钮：`TRAIN_BS`、`rollout.agent.num_workers`。
  2. **墙钟时间**：step 时长随 F 近线性（131→242 s，F=64→256，同批量），
     随 K/TRAIN_BS 线性。150 步生产跑的总时长按此外推。
- **帧数选择**：若维持 PLAN §6 的 64 帧，一切现值即可。若要提高（更高
  的证据覆盖率），**F=128 或 192 是甜点位**：GPU/RAM 都无压力，只多付
  23%/60% 的步时；需要同步做的事——
  1. `constants.GLOBAL_NUM_FRAMES` 改值 → prompts 文本变 → 按 PLAN §4
     重渲染 SFT 数据 + 重跑 extract_rl.py（字节一致纪律）；
  2. `run_sft.sh` 的 `GLOBAL_FRAMES` 与 RL 的 `MAX_PROMPT_LEN` 按 §1 表
     联动（SFT 侧 f96–f192 smoke 早已通过，显存 33–36 GB 平坦）；
  3. F>446 还需 extract_rl.py 加逐视频帧数封顶。
- **K**：对 GPU 免费；成本=步时线性 + RAM（轨迹数×pixel tensor）。K=16
  在 F≤128 下 RAM 可行（外推），但先想清楚训练总时长预算。
- **crop 帧数（C）**：C>16 吃 response 预算（§1 表），C=32 需要
  RESP 16384 + limit_images 112 + token 上限 24576+。GPU 侧预计同样宽松，
  待 C=32 smoke 验证。
- **遗留观察**：每次 run 结束时偶见 1 个 DataLoader worker 被 SIGKILL
  （teardown 期预取），不影响训练与退出码；若在 step 中间出现则是 RAM
  预警，优先降 `data.dataloader_num_workers`。

## 4.5 定版生产配置（2026-08-26，用户拍板）

```
全局视图  F=128 @ ≤50176 px/帧（constants.GLOBAL_NUM_FRAMES）
crop     C=30 固定 @ ≤150528 px（≈论文 1 fps 的典型密度；constants.CROP_NUM_FRAMES）
GRPO     K=16 · TRAIN_BS=16（=256 轨迹/步，150 步 ≈ 52 h）
预算     MAX_PROMPT_LEN=4608 · MAX_RESP_LEN=16384 · max_model_len 20992
执行层   GPU_MEM_UTIL=0.65 · ppo/log_prob 装箱 24576 · max_num_batched_tokens 24576
         · limit_images=112 · engine_kwargs.vllm.mm_processor_kwargs.max_pixels=150528
           （防 vLLM 用 preprocessor 默认 16.7M px/图造 profiling 假数据吃光 KV 池）
SFT 侧   GLOBAL_FRAMES=128 · data.max_length=20480 · max_token_len_per_gpu=24576
数据     judge 时代 allocation：SFT 600 题（全加权采样）/ RL 1,068 · 零丢弃
         （--max-gt-words 默认 999；见 DATA.md §3）
```

依据：§2 表全部通过项 + C=30 预算验证 + 证据窗口覆盖率（64 帧 12.4% 的问题
窗口内 <3 帧 → 128 帧降到 1.7%）+ selftrace 实测论文 crop 为 1 fps。
C=30 与"1 fps 变帧数"两方案中用户选定固定 30（schema/预算形状不变）。

## 5. 复现命令

```bash
# 任一档位（示例 F=192）：
EXP_NAME=grpo_smoke_f192 TRAIN_FILE=data/processed/rl_train_f192.parquet \
MAX_PROMPT_LEN=6144 TRAIN_BS=8 GROUP_SIZE=4 TOTAL_STEPS=2 \
bash run_grpo.sh trainer.val_before_train=False trainer.save_freq=-1 trainer.test_freq=-1
# 帧数变体 parquet 由 rl_train.parquet 改写 videos[].nframes 生成；
# F>446 需按视频总帧数封顶（见 §3.1）。
```
