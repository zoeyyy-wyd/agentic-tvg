# Agentic-TVG 项目方案

**基于 Qwen3-VL + verl 的多轮工具调用视频时序定位（Temporal Video Grounding）**

版本：v1.0 ｜ 日期：2026-08-18 ｜ 状态：待启动

---

## 1. 项目定位

### 1.1 一句话描述

在纯时序定位（TVG）任务上研究 agentic 多轮工具调用 + 可验证奖励（RLVR）的作用：模型通过 `crop_video(start_time, end_time)` 工具对视频进行"全局粗看 → 局部细查 → 修正"的循环，最终输出目标事件的时间区间 `[t_s, t_e]`。

### 1.2 与 LongVT 的关系

LongVT（arXiv:2511.20785）的主任务是长视频开放式 QA，TVG 只是其 reward 的一个分量，且全部结论建立在 Qwen2.5-VL 上。本项目做两件 LongVT 没做的事：

1. **任务分解**：把 agentic 范式收缩到纯 TVG 上。TVG 的 ground truth 是时间区间，奖励可完全由 IoU 验证，不需要 LLM-as-a-Judge，是标准 RLVR 设定，适合单卡训练。
2. **结论重检**：LongVT 在 Qwen2.5-VL 上得出的三个结论——(a) 冷启动 SFT 不可省，(b) tool reward 不必要，(c) IoU 奖励优于 Recall——在原生工具调用和时序定位能力大幅增强的 Qwen3-VL 上是否仍然成立，无人验证过。本项目以此为核心研究问题。

### 1.3 设计原则

- **零数据构造**：训练数据全部来自 LongVT 官方发布（HuggingFace: `longvideotool/LongVT-Parquet`），评测用标准 benchmark。工程预算全部投入训练策略与奖励设计。
- **单卡可行**：所有配置以 1× A100 80GB（COSMOS srv4-lg2）为约束设计。
- **不做跨代对照**：不与 Qwen2.5-VL 系对比，以自测的 zero-shot 基线为锚点，结果以相对提升报告。

---

## 2. 技术栈

| 组件 | 选型 | 说明 |
|---|---|---|
| 基座模型 | **Qwen3-VL-4B-Instruct** | 不用 Thinking 版（多轮 rollout 中 reasoning content 的 chat template 处理是 verl 文档明示的坑）；不用 2B（grounding 能力不足） |
| RL 框架 | **verl 官方最新 release** | 多轮工具调用走 SGLang rollout 路径（verl 多轮支持的一等公民）；不用 LongVT fork（绑死 Qwen2.5-VL） |
| Rollout 引擎 | SGLang（最新版） | 与 policy 同卡部署 |
| SFT 框架 | LLaMA-Factory（首选）或 verl FSDP SFT trainer | LLaMA-Factory 对 Qwen3-VL 适配通常最快 |
| 依赖版本 | transformers ≥ 4.57，verl main 对应 pin | 以 verl 仓库 requirements + 自带 multiturn VLM example（geo3k）跑通为准 |
| 环境 | srv4-lg2 新建 conda env（如 `agentic_tvg`） | 不动现有 `trl` / `dpo` 环境 |

**工具接口**（沿用 LongVT 最小设计）：

- 单一工具 `crop_video(start_time: float, end_time: float)`，返回该区间重采样的帧
- 交互格式：`<think>` → `<tool_call>` → `<tool_response>` → 再 think →（可再调用）→ `<answer>[t_s, t_e]</answer>`
- 最大轮数 T = 3
- Tool call 按 Qwen3-VL 原生 hermes 风格格式

**帧与 token 预算**（单卡命门，双头卡死：system prompt 声明 + processor 配置强制）：

- 全局粗看：32 帧，低分辨率（低 `min_pixels`）
- 每次 crop 返回：16 帧，较高分辨率
- 总 context ≤ 12K tokens

---

## 3. Step 0 — Zero-shot Agentic 探针（训练前必做）

**目的**：Qwen3-VL 原生能力已强，冷启动 SFT 从"必需品"变为"待定剂量的变量"。用纯推理实验确定剂量，同时产出报告第一张表。

**设定**：SGLang 起 Qwen3-VL-4B-Instruct，挂 crop_video 工具，在 Charades-STA + 一个长视频子集上测三种模式：

| 模式 | 描述 |
|---|---|
| (a) Direct | 无工具，直接输出时间区间 |
| (b) Tool-optional | prompt 提供工具，不强制调用 |
| (c) Tool-forced | 强制至少调用一次 |

**观测指标**：mIoU、工具调用率、调用后 window 修正率（第二次 proposal 相对第一次的 IoU 变化）。

**决策规则**：

- 调用率 > 50% 且格式基本正确 → 轻量 SFT（2K traces）
- 调用行为混乱 / 格式崩坏 → 全量 SFT（6.4K traces）

---

## 4. 数据方案（零构造）

### 4.1 来源

| 用途 | 数据 | 规模 | 工作量 |
|---|---|---|---|
| SFT 冷启动 | LongVT-Parquet 中 Qwen-distilled temporal grounding iMCoTT traces | 6,395 条 | 重渲染脚本（见 4.2） |
| RL 训练 | LongVT-Parquet RL split 的 (video, query, GT window) 三元组 | ~1.6K，从 grounding 子集补到 3–5K | 抽取脚本（几十行） |
| 评测 | Charades-STA（官方 AllenAI 包直接下载）+ 从 LongVT 数据切分的长视频 held-out 集 | — | 下载即用 |

**明确不用**：ActivityNet-Captions（视频需从 YouTube 抓取，失效率高）；LongVT 的 22 万条 non-tool CoT（服务于通用 QA，与本任务无关）。

### 4.2 唯一的真实数据工作：traces 重渲染

LongVT 的 SFT traces 按 Qwen2.5-VL chat template 存储。需写脚本抽取 `<think>` / `<tool_call>` / `<tool_response>` 内容，按 Qwen3-VL template（hermes 风格 tool call）重新渲染。纯文本变换，约一天工作量。**必须做**——template 不匹配的 SFT 会破坏模型原生工具调用能力。

### 4.3 存储

视频文件：LongVT repo 下载脚本 + Charades 官方包，服务器预留 150–200 GB。

---

## 5. 训练流程

### Stage 1 — SFT 冷启动（剂量由 Step 0 决定）

- LoRA r=16，剂量 2K 或 6.4K traces
- 训练目标仅三项：会提 window、会读 crop 返回的帧、window 错误时会修正
- 预计半天至两天（单卡）

### Stage 2 — GRPO（核心阶段）

**奖励函数**（verl function-based reward，sandbox 内 python 函数）：

```python
def compute_reward(pred_span, gt_span, format_ok):
    r_fmt = 0.5 if format_ok else 0.0
    iou = temporal_iou(pred_span, gt_span)
    return r_fmt + iou   # 消融时替换为 penalty-aware 版本
```

**单卡配置**：

| 项 | 值 |
|---|---|
| 训练方式 | LoRA RL（policy + SGLang engine 同卡） |
| gpu_memory_utilization（rollout 侧） | ~0.45 |
| Group size | 8 |
| Train batch | 32–64 prompts |
| Max turns | 3 |
| Temperature | 1.0 |
| 步数 | 100–200，reward 饱和即停 |

**难度过滤**：训前离线 K=8 rollout，剔除全对/全错样本（zero-variance group 处理，同 GSM8K/GRPO 经验）。TVG 任务中一组 rollout 全部 IoU≈0 的情况会比数学题更常见，此步不可省。

### Stage 3 — RFT 自蒸馏（可选）

筛选 IoU ≥ 0.5 的 RL rollout 轨迹回灌 SFT。时间不足则砍掉，写入 future work。

---

## 6. 消融实验（挑两个执行）

1. **冷启动剂量**：0（纯 RL）vs 2K vs 6.4K traces
   → 核心贡献：直接回答"强 base 是否仍需要 LongVT 式冷启动"
2. **奖励形状**：vanilla IoU vs penalty-aware IoU（对严重超长预测区间额外惩罚，抑制 span inflation；先验来自 VideoTemp-o3）
3. （备选）**多轮价值**：T=3 vs T=1，按视频时长分桶报告
   → 预期 claim：视频越长、目标区间占比越小，多轮相对增益越大（duration-stratified 分析，LongVT 论文中没有的图）

---

## 7. 评测方案

- **Charades-STA**：R@0.3 / R@0.5 / R@0.7、mIoU（标准 TVG 指标）
- **长视频 held-out**（自 LongVT 数据切分）：同上指标 + 按时长分桶
- **过程指标**：工具调用率、平均轮数、window 修正成功率
- **锚点**：Step 0 的 zero-shot 三模式基线；所有训练结果报告为相对提升

---

## 8. 时间线

| 周次 | 内容 |
|---|---|
| 第 1 周（与 DUCB-Acc 并行，碎片时间） | 新建环境；verl 自带 multiturn VLM example 跑通；Step 0 探针；traces 重渲染脚本；视频下载启动 |
| 第 2 周 | SFT 冷启动；评测 pipeline 完成；zero-shot 基线定稿 |
| 第 3–4 周（9 月初，summer report 交付后） | GRPO 主实验 + 两个消融 |
| 第 5 周（9 月中） | （可选）RFT；README / 技术报告；写入简历 |

---

## 9. 风险清单

| 风险 | 应对 |
|---|---|
| verl 的 LoRA + SGLang + video + tool 完整组合未必有人踩过 | 预留一周 debug 预算；卡住时在 verl issue 区搜 `qwen3_vl` 关键词；第 1 周先跑官方 example 验证链路 |
| Qwen3-VL 的 DeepStack 结构与新 processor 显存曲线不同于上代 | 帧数上限以实测为准，从 32/16 预算起步向上探 |
| 强 base 压缩 agentic delta | Step 0 探针提前暴露；若 zero-shot 已很强，叙事转向"消融揭示冷启动/多轮在强 base 上的边际价值"——负结果同样有信息量 |
| chat template 不匹配毁掉工具调用 | 重渲染脚本产出后先小批量 SFT + 人工检查输出格式，再全量 |
| 无跨代对照数字 | 以自测 zero-shot 基线为锚，全部报告相对提升 |

---

## 10. 简历与面试叙事

- **一句话**：Agentic Temporal Video Grounding with Verifiable Rewards — 基于 Qwen3-VL-4B + verl，通过多轮 crop_video 工具调用与 IoU 可验证奖励的 GRPO 训练，在单卡上系统检验了冷启动剂量、奖励形状与多轮交互对时序定位的贡献。
- **与 DUCB-Acc 的配对**：一个是 preference 数据侧的 bandit 调度（DPO 线），一个是 agentic RL 的奖励设计与多轮 rollout（GRPO 线），共同覆盖 LLM post-training 岗位的两条主干。
- **数据策略的说法**："工程预算全部押在训练策略和奖励设计上，数据用开源发布"——对 post-training 算法岗是加分表述。
