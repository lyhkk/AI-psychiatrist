# CBT-Discover：认知行为治疗多智能体辅助系统

## 项目概述

CBT-Discover（Cognitive Behavioral Therapy Discovery System）是一个基于 LangGraph 的多智能体心理辅助研究系统。系统将「临床诊断」与「对话干预」彻底解耦，通过隐式思维链（MDP-CoT）提升临床保真度，并引入患者模拟器与信息熵评测实现全自动的科学量化评估。

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                  沙盘模拟层 run_simulation.py         │
│  ┌──────────────┐          ┌──────────────────────┐  │
│  │ PatientNode  │◄────────►│  干预工作流（LangGraph）│  │
│  │ 患者模拟器   │          │  ┌──────────────────┐ │  │
│  │ PATIENT_*    │          │  │ DiagnosticianNode│ │  │
│  │ (或 SUPERVISOR_*)│      │  │  后台诊断追踪器   │ │  │
│  └──────────────┘          │  │  SUPERVISOR_*    │ │  │
│                             │  └────────┬─────────┘ │  │
│                             │           │ CBTForm    │  │
│                             │  ┌────────▼─────────┐ │  │
│                             │  │  TherapistNode   │ │  │
│                             │  │  MDP-CoT 治疗师  │ │  │
│                             │  │  THERAPIST_*     │ │  │
│                             │  └──────────────────┘ │  │
│                             └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ transcript JSON
┌─────────────────────────────────────────────────────┐
│               评估层 eval_pipeline.py                │
│  模块一：IG-PQA 事实证据清晰度信息增益                │
│  模块二：CTRS 临床保真度评分（LLM-as-a-Judge）        │
│  模块三：Belief Conviction 信念确信度衰减             │
│                        JUDGE_*                       │
└─────────────────────────────────────────────────────┘
```

---

## 核心模块说明

### 1. 智能体节点（`agents/`）

#### `DiagnosticianNode`（`diagnostician.py`）
- **角色**：后台静默节点，对话中不向患者发言。
- **职责**：每轮对话后分析最新上下文，提取并更新 CBT 认知评估表（`cbt_form`）中的四个字段：情境、情绪、自动思维、认知扭曲类别。
- **输出**：严格 JSON 格式，经 Pydantic 校验后写入全局状态。
- **模型配置**：读取 `.env` 中的 `SUPERVISOR_*` 变量（推荐高推理能力模型）。

#### `TherapistNode`（`therapist.py`）
- **角色**：前端对话治疗师（MDP-CoT）。
- **职责**：在 `DiagnosticianNode` 更新表单后运行，先在 `<inner_monologue>` 中完成三步规划（防御评估 → 表单缺口分析 → 策略选择），再在 `<response>` 中输出自然的中文回复。
- **输出格式**：XML 双标签结构，由 `_parse_xml_output()` 严格解析分离。
- **模型配置**：读取 `.env` 中的 `THERAPIST_*` 变量（前端对话模型）。

#### `PatientNode`（`patient.py`）
- **角色**：患者模拟器（仅用于沙盘评测，真实部署时不启用）。
- **职责**：扮演高防御型来访者，对生硬说教或无底线迎合会产生抵触，仅对深度共情与苏格拉底式引导作出积极回应。
- **初始化**：接收来自 PsyQA 数据集的 `question` + `description` 作为角色背景剧本。
- **模型配置**：优先读取 `PATIENT_*`，回退到 `SUPERVISOR_*`。

#### `LLMClient`（`llm_base.py`）
- 统一的大模型调用客户端，兼容所有 OpenAI Chat Completions 格式的 API（DeepSeek / GLM / GPT-4o 等）。
- 通过工厂方法 `LLMClient.from_role(role)` 按角色自动读取 `.env` 配置。
- 提供 `chat()`、`simple_chat()`、`extract_json()` 三个核心方法。

#### `build_intervention_graph()`（`workflow.py`）
- 使用 LangGraph `StateGraph` 构建并编译干预工作流。
- 图结构：`[入口] → DiagnosticianNode → TherapistNode → [END]`
- 每次 `invoke()` 即执行一个完整的「诊断 + 治疗」轮次。

#### `DialogueState` / `CBTForm`（`state.py`）
- 全局共享状态，所有节点通过读写此状态交互，不直接传递消息。
- 核心字段：`chat_history`、`cbt_form`、`entropy_scores`、`last_patient_msg`、`last_therapist_response`、`last_inner_monologue`、`turn_count`。

---

### 2. 运行模式（`run_simulation.py`）

支持两种运行模式，输出的 transcript JSON 格式完全相同，可直接送入评估管线：

| 模式 | 说明 | 模型配置 |
|------|------|----------|
| `cbt-discover` | DiagnosticianNode + TherapistNode 双 Agent 系统 | `SUPERVISOR_*` + `THERAPIST_*` |
| `baseline` | 单一通用大模型咨询师，无后台诊断，用于对比实验 | `BASELINE_*` |

**输出 transcript 结构：**
```json
{
  "meta": { "mode", "psyqa_index", "turns_completed", "timestamp", ... },
  "final_cbt_form": { "situation", "emotion", "automatic_thought", "cognitive_distortion" },
  "transcript": [
    { "role": "patient",   "content": "...", "turn": 0 },
    { "role": "therapist", "content": "...", "inner_monologue": "...", "cbt_form_snapshot": {...}, "turn": 1 },
    ...
  ]
}
```

---

### 3. 评估管线（`eval_pipeline.py`）

评估管线共三个模块，全部通过 `.env` 中的 `JUDGE_*` 模型执行。

#### 模块一：事实证据清晰度信息增益（IG-PQA）
- **测算标的**：患者对自身负面情绪提供的**客观事实证据的清晰程度**，而非认知扭曲分类。
- **原理**：苏格拉底提问的目的是把患者从「模糊的宏观抱怨」拉回到「清晰的微观事实」。通过测量事实细节的熵减，CBT-Discover 的高价值提问将获得显著正向 IG。
- **计算方式**：
  1. 治疗师提问**前**，LLM 输出「事实证据清晰度」5 维度概率分布，计算香农熵 H(before)。
  2. 患者回答**后**，再次计算香农熵 H(after)。
  3. 信息增益 IG = H(before) − H(after)，IG > 0 记为一次高价值苏格拉底提问。
- **5 维度定义**：

  | 维度 | 含义 |
  |------|------|
  | 完全模糊 | 仅有宏观抱怨或情绪宣泄，无具体事实 |
  | 轻度具体 | 提及大致情境，缺少时间/地点/后果 |
  | 中度具体 | 有明确事件和部分细节，但因果链不完整 |
  | 高度具体 | 完整事件描述，含时间、地点、经过、客观后果 |
  | 反事实澄清 | 来访者主动区分主观解读与客观事实，认知开始松动 |

- **输出指标**：`ig_mean`（平均信息增益，bits）、`ig_positive_ratio`（高价值提问占比）。

#### 模块二：临床保真度评分（CTRS）
- **原理**：基于认知疗法评定量表（Cognitive Therapy Rating Scale），由 LLM 担任裁判（LLM-as-a-Judge），对整段对话整体评分。
- **评分维度**（0–6 分，0=破坏性，3=合格，6=杰出）：
  - `understanding`：理解与共情——是否准确捕捉患者内部现实，避免虚假迎合？
  - `guided_discovery`：引导式发现——是否有效使用苏格拉底提问引发认知顿悟，而非直接说教？
  - `interpersonal_effectiveness`：人际效能——是否展现真实专业性并成功化解高防御状态？
- **输出指标**：三维度得分、`ctrs_avg` 综合均值、`justification`（引用对话原句的判分依据）。

#### 模块三：信念确信度衰减（Belief Conviction Decay）
- **测算标的**：患者对核心负面信念（如「我是个废物」「我永远不会成功」）的**确信程度**（0–100）。
- **原理**：这是目前评测多轮 CBT 治疗最前沿的量化方法。有效的苏格拉底提问应使患者产生犹豫，确信度下降；激起防御则确信度上升。
- **计算方式**：
  - 对每个 therapist→patient 轮次：Score = Conviction(提问前) − Conviction(回答后)
  - Score > 0：信念松动（高价值）；Score < 0：防御激活（负向）
- **评分锚点**：

  | 分值 | 含义 |
  |------|------|
  | 100 | 深信不疑，完全封闭，不接受任何质疑 |
  | 75 | 坚定持有，偶有犹豫但立刻收回 |
  | 50 | 有所动摇，能听进部分质疑 |
  | 25 | 开始怀疑，开放度明显提升 |
  | 0 | 已完全放弃核心负面信念 |

- **输出指标**：`conviction_start`（初始确信度）、`conviction_end`（最终确信度）、`total_decay`（总衰减量）、`delta_mean`（每轮平均衰减）、`positive_ratio`（有效松动轮次占比）、逐轮详情列表。

**评估报告输出结构（JSON）：**
```json
{
  "meta": { ... },
  "final_cbt_form": { ... },
  "ig_pqa": {
    "ig_mean_bits": 0.32,
    "ig_positive_ratio": 0.7,
    "high_value_turns": 7,
    "total_therapist_turns": 10,
    "ig_list": [ ... ]
  },
  "ctrs": {
    "understanding": 5,
    "guided_discovery": 4,
    "interpersonal_effectiveness": 5,
    "ctrs_avg": 4.67,
    "justification": "..."
  },
  "belief_conviction": {
    "conviction_start": 90,
    "conviction_end": 55,
    "total_decay": 35,
    "delta_mean": 3.5,
    "positive_ratio": 0.6,
    "effective_turns": 6,
    "total_therapist_turns": 10,
    "conviction_list": [ ... ]
  }
}
```

---

## 数据集

| 数据集 | 路径 | 用途 |
|--------|------|------|
| PsyQA | `datasets/PsyQA/PsyQA_full.json` | 为患者模拟器提供真实心理咨询背景剧本 |
| CBT-Bench | `datasets/CBT-Bench/` | CBT 相关分类与诊断能力基准测试 |
| SupervisedVsLLM-EfficacyEval | `datasets/SupervisedVsLLM-EfficacyEval/` | 监督学习与 LLM 效果对比参考数据 |

---

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 配置 `.env`

在项目根目录创建 `.env` 文件，按角色配置各模型的 API 信息：

```dotenv
# 前端对话治疗师（TherapistNode）
THERAPIST_API_KEY=your_key
THERAPIST_BASE_URL=https://api.example.com/v1
THERAPIST_MODEL=your-model-name

# 后台临床诊断器（DiagnosticianNode）
SUPERVISOR_API_KEY=your_key
SUPERVISOR_BASE_URL=https://api.example.com/v1
SUPERVISOR_MODEL=your-model-name

# 评测裁判模型（eval_pipeline.py）
JUDGE_API_KEY=your_key
JUDGE_BASE_URL=https://api.example.com/v1
JUDGE_MODEL=your-model-name

# 对比基线单模型（baseline 模式）
BASELINE_API_KEY=your_key
BASELINE_BASE_URL=https://api.example.com/v1
BASELINE_MODEL=your-model-name

# 通用回退默认值（可选）
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

> 各节点按角色优先读取对应前缀变量，缺失时自动回退到 `LLM_*` 通用变量。

---

## 快速开始

### 标准流程（CBT-Discover 双 Agent 系统）

```bash
# Step 1：生成对话记录（transcript）
python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0

# 模拟多个对话
--psyqa-index 0 1 2        # 多个单值
--psyqa-index 0-4          # 闭区间 → [0,1,2,3,4]
--psyqa-index 0-2 5 8-9    # 混合写法 → [0,1,2,5,8,9]


# Step 2：评估对话质量
# 单文件
python eval_pipeline.py --transcript results/sim/cbt-discover/psyqa0_xxx.json

# 多文件
python eval_pipeline.py --transcript results/sim/cbt-discover/psyqa0_xxx.json results/sim/cbt-discover/psyqa1_xxx.json

# 整个目录
python eval_pipeline.py --transcript results/sim/cbt-discover/

# 指定输出目录
python eval_pipeline.py --transcript results/sim/cbt-discover/ --output-dir results/eval/cbt-discover/
```

### 对比实验（CBT-Discover vs Baseline）

```bash
# 运行 CBT-Discover 双 Agent 系统
python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0 --output results/sim_cbt.json

# 运行单模型基线（相同患者背景）
python run_simulation.py --mode baseline --turns 10 --psyqa-index 0 --output results/sim_baseline.json

# 分别评测，对比两份报告
python eval_pipeline.py --transcript results/sim_cbt.json     --output results/eval_cbt.json
python eval_pipeline.py --transcript results/sim_baseline.json --output results/eval_baseline.json
```

### CLI 参数说明

**`run_simulation.py`**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `cbt-discover` | 运行模式：`cbt-discover` 或 `baseline` |
| `--turns` | `10` | 最大对话轮数 |
| `--psyqa` | `datasets/PsyQA/PsyQA_full.json` | PsyQA 数据集路径 |
| `--psyqa-index` | `0` | 使用 PsyQA 第几条记录作为患者背景 |
| `--output` | 自动生成 | 输出 JSON 路径 |

**`eval_pipeline.py`**

| 参数 | 说明 |
|------|------|
| `--transcript` | （必填）`run_simulation.py` 生成的 transcript JSON 路径 |
| `--output` | 评估报告输出路径（默认自动生成至 `results/` 目录） |

---

## 评估报告示例输出

```
════════════════════════════════════════════════════════════
  CBT-Discover 评估报告  |  PsyQA #0
════════════════════════════════════════════════════════════
  对话轮数         : 10
  最终认知扭曲     : 非此即彼（全或无思维）
  最终情绪         : 绝望、无力感

  【IG-PQA 事实证据清晰度增益评测】
  平均 IG          : 0.3142 bits
  高价值提问比例   : 70.0%  (7/10 轮)

  【CTRS 临床保真度评分 (0-6)】
  理解与共情       : 5 / 6
  引导式发现       : 4 / 6
  人际效能         : 5 / 6
  综合平均         : 4.67 / 6

  判分依据: 咨询师多次使用「你说的'永远'是指...」类苏格拉底提问...

  【信念确信度（Belief Conviction）衰减指标】
  初始确信度       : 90 / 100
  最终确信度       : 55 / 100
  总衰减量         : +35 分
  每轮平均衰减     : +3.5 分
  有效松动轮次     : 6/10 轮  (60.0%)
════════════════════════════════════════════════════════════
```

---

## 项目文件结构

```
CBT-newer/
├── agents/
│   ├── __init__.py
│   ├── diagnostician.py   # 后台临床循证追踪器
│   ├── llm_base.py        # 通用 LLM 调用客户端
│   ├── patient.py         # 患者模拟器（沙盘专用）
│   ├── state.py           # 全局共享状态定义
│   ├── therapist.py       # MDP-CoT 治疗师
│   └── workflow.py        # LangGraph 干预工作流
├── datasets/
│   ├── CBT-Bench/         # CBT 基准测试数据
│   ├── PsyQA/             # 心理咨询问答数据集
│   └── SupervisedVsLLM-EfficacyEval/
├── results/               # 模拟与评估输出（自动生成）
├── cache/                 # LLM 调用缓存
├── run_simulation.py      # 沙盘模拟主入口
├── eval_pipeline.py       # 评估管线主入口
├── requirements.txt
└── .env                   # API 密钥配置（需自行创建）
``` 