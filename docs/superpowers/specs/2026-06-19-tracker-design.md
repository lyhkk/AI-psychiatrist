# CBT-Discover · Tracker（进展追踪器）设计 Spec

- 日期：2026-06-19
- 状态：待评审（未提交）
- 作者：lyhkk + Claude
- 关联：建立在已完成的「会话持久化 + 历史/继续/撤销 + 下载」之上（Feature A+B）。

## 1. 目标

提供一个**进展追踪器**：以一段过去的会话为基线，针对当时识别出的认知问题，发起一次**互动式复诊（check-in）**，对比两次状态、量化用户是否好转，并沿时间轴呈现趋势。

Tracker 与「继续对话（resume）」本质不同：
- resume = 在原会话上**接着聊**。
- tracker = 做一次**新的复诊**，计算两次表现的 **diff**，判断是否变好。

## 2. 范围

### In scope
- 互动式复诊：Tracker 据基线识别出的问题逐一重探用户。
- 进展判定：以**信念确信度（Belief Conviction, 0–100）**为主指标，叠加 CBT 诊断表（情绪/自动思维/认知扭曲）的变化与 LLM 对比叙述。
- 报告：逐维度 diff 表 + 确信度趋势图（多次复诊连线）+ 整体状态 + 叙述。
- 独立「追踪」页面（SPA 内视图切换）。

### Out of scope（v1 明确不做）
- 复诊中的自适应深问（仅做有界的、模板化的定向重探）。
- 任意轮回溯 / 分叉（属 Feature B 范畴，已确定不做）。
- 修改任何既有 prompt 或 agent 行为。
- 新增 `.env` 角色（复用 `THERAPIST_*` / `JUDGE_*` / `SUPERVISOR_*`）。

## 3. 硬性约束（信源接地）

> 用户明确要求：新 prompt 必须有充足的认知行为学信源支撑，不得自创无意义 prompt；不得改动已有 prompt。

- **确信度打分**：原样复用 `eval_pipeline.get_belief_conviction()`（含 0–100 临床确信度量表，见 [eval_pipeline.py:369](../../../eval_pipeline.py)）。
- **认知扭曲分类**：复用 `diagnostician.COGNITIVE_DISTORTIONS`（12 类标准分类法）与 `DiagnosticianNode` 对复诊重新诊断。
- **复诊问句**：**模板兜底 + LLM 受限润色**。确定性模板是唯一信源（仅就基线已识别字段做"信念再评分 / 自动思维再检验 / 情绪再评估 / 应对回顾"——标准 CBT 随访技术）；随后用 LLM 在**严格约束**下仅润色措辞使其更自然：不得改变含义、不得新增话题/建议、必须保留模板引用的基线原文与 0–100 评分要求。润色结果须通过校验（§10.1），不通过则回退原模板 → 杜绝自创。
- **对比叙述**：judge 模型，输入为算好的确信度分数 + 诊断表 diff + 证据，仅做总结，禁止编造未提供的信息。
- **不触碰** `therapist.py` / `diagnostician.py` 的既有 prompt；Tracker 全部为新增、独立模块。

## 4. 概念模型

- **基线 baseline**：一段已有会话（`results/sessions/<id>.json`），含其最终 `cbt_form` 与 `chat_history`。
- **复诊 check-in**：一次新的、有界的互动，存于 `results/trackers/<checkin_id>.json`，通过 `baseline_id` 链接基线。
- **报告 report**：复诊完成后计算，内嵌于复诊记录的 `report` 字段。
- **趋势 trend**：同一 baseline 下，基线 + 历次复诊的确信度时间序列。

## 5. 架构与组件（全部加法式）

| 层 | 文件 | 职责 |
|---|---|---|
| Agents | `agents/tracker.py`（新增） | `TrackerNode`：据基线 `cbt_form` 用确定性模板生成有序复诊问句，再经 LLM（`THERAPIST_*` 模型）受限润色措辞，校验失败回退模板 |
| Eval（复用） | `eval_pipeline.py` | `get_belief_conviction(llm, text)` 直接复用做确信度打分 |
| Agents（复用） | `agents/diagnostician.py` | `DiagnosticianNode` 对复诊对话重新诊断，得当前 `cbt_form` |
| Web core | `webapp/core/tracker_service.py`（新增） | 编排：加载基线 → 推进复诊 → 计算报告；用 `SessionStore(results/trackers)` 持久化 |
| Web core（复用） | `webapp/core/persistence.py` | `SessionStore` 指向 `results/trackers` 目录复用 |
| Web routes | `webapp/routes/tracker.py`（新增蓝图 `/api/tracker/*`） | HTTP 编解码，委托 `TrackerService` |
| 前端 | `index.html` / `app.js` / `style.css` | 「追踪」视图：基线选择 → 复诊对话 → 报告（diff 表 + SVG 趋势图 + 叙述） |

模块边界：`TrackerNode` 纯生成问句、无副作用；`TrackerService` 编排并依赖 agents/eval 作为库；routes 仅做 HTTP；前端仅做展示。各单元可独立测试。

## 6. 复诊流程（有界）

1. 用户在「追踪」页选定一个基线 → `POST /api/tracker/start`。
2. `TrackerService` 加载基线记录，读取 `baseline_form`，调用 `TrackerNode.build_questions(baseline_form)` 得到有序问句（仅针对非空维度）。新建复诊记录（`status="in_progress"`, `q_index=0`），返回第 1 问。
3. 用户作答 → `POST /api/tracker/message {checkin_id, message}`：把答案追加进 `chat_history`，`q_index += 1`。若还有问句，返回下一问；否则进入第 4 步。
4. 全部问句答完 → `TrackerService.compute_report(checkin_id)`：计算报告，写入 `report`，`status="completed"`，返回 `{done:true, report}`。

问句集（确定性，按基线非空字段顺序生成；`{...}` 为基线字段插值）：
1. **情境再遇**（若 `situation`）：`上次你提到在「{situation}」时感到困扰。最近有没有再遇到类似的情境？当时发生了什么？`
2. **自动思维 + 确信度再评分**（若 `automatic_thought`）：`那时你脑海里会冒出「{automatic_thought}」这个想法。最近再遇到类似情况时，这个想法还会出现吗？如果用 0–100 表示你现在有多相信它（100=完全相信，0=已不相信），你会打几分？`
3. **情绪再评估**（若 `emotion`）：`再遇到这种情况时，你的情绪和上次的「{emotion}」相比，有什么变化吗？`
4. **应对回顾**（始终）：`这段时间，你有没有尝试一些新的方式去应对它？效果怎么样？`

问句生成 = **模板兜底 + LLM 受限润色**：先按上面模板填充基线字段（确定性、唯一信源），再对每条做一次受限 LLM 润色（仅改措辞，见 §10.1 的 prompt 与校验），润色失败/不过校验则用原模板。

> 有界性：问句数 = 非空维度数（≤4）。润色为每条 1 次 LLM 调用（≤4 次，可经 `ENABLE_TRACKER_POLISH=false` 关闭退回纯模板），成本可控、可测。自适应深问列为 future work。

## 7. 报告计算

输入：`baseline_form`、`baseline.chat_history`、复诊 `chat_history`。

### 7.1 确信度（主指标）
- `baseline_conviction, baseline_evidence = get_belief_conviction(judge_llm, format_text(baseline.chat_history))`
- `current_conviction, current_evidence = get_belief_conviction(judge_llm, format_text(checkin.chat_history))`
- `conviction_delta = baseline_conviction - current_conviction`（正 = 确信度下降 = 好转）
- `format_text()`：把 `{role,content}` 列表渲染为 `来访者：…/咨询师：…` 文本（user→来访者，assistant→咨询师），适配 `get_belief_conviction` 的纯文本入参。
- **自评分 vs LLM 估分**：§6 第 2 问中用户自报的 0–100 只作为复诊文本的一部分（增强信号），`current_conviction` 仍由 `get_belief_conviction` 对复诊文本统一估算——与基线同一口径，保证两侧可比。

### 7.2 整体状态（阈值）
- `conviction_delta >= 15` → `改善`
- `conviction_delta <= -15` → `恶化`
- 否则 → `持平`

### 7.3 诊断表 diff
- `current_form = DiagnosticianNode()(state_from(checkin.chat_history))["cbt_form"]`
  - `state_from()`：构造一个 `DialogueState`，`chat_history` = 复诊对话，**`cbt_form` 置空**（`make_initial_state()` 的空表）。从空表开始 → 当前诊断只反映本次复诊暴露的信息，不被基线值污染，从而能真实反映"变化"。
- 对 `[emotion, automatic_thought, cognitive_distortion, situation]` 逐维度产出 `{dimension, before, after, change}`：
  - 两者非空且相等 → `same`
  - 两者非空且不等 → `changed`
  - before 非空、after 为空 → `not_reassessed`（本次未重探到，**不**判定为"消失"，避免误信号）
  - before 为空、after 非空 → `new`
  - `cognitive_distortion` 为 `same` = 模式仍在；`changed` = 认知模式转移（叙述层解释，不等同于好转）。

### 7.4 趋势
- 聚合该 `baseline_id` 下所有 `completed` 复诊：
  `trend = [{label:"基线", date: baseline.created_at, conviction: baseline_conviction}, {label:"复诊1", date, conviction}, …]`
- 图注：确信度**下降为好转**。

### 7.5 叙述（judge 模型，接地）
单次 LLM 调用，输入为上面算好的分数 + 证据 + 表 diff，输出 `{status, narrative, suggestion}`。Prompt 草案见 §10.2。

## 8. 数据模型 `results/trackers/<checkin_id>.json`

```json
{
  "checkin_id": "uuid4",
  "baseline_id": "<session_id>",
  "kind": "tracker_checkin",
  "status": "in_progress | completed",
  "created_at": "ISO-8601",
  "last_active": "ISO-8601",
  "model": "<JUDGE/SUPERVISOR model>",
  "baseline_form": { "situation": null, "emotion": null, "automatic_thought": null, "cognitive_distortion": null },
  "questions": ["…", "…"],
  "q_index": 0,
  "chat_history": [{ "role": "assistant|user", "content": "…" }],
  "report": {
    "baseline_conviction": 0, "baseline_evidence": "",
    "current_conviction": 0, "current_evidence": "",
    "conviction_delta": 0,
    "status": "改善|持平|恶化",
    "current_form": { "...": null },
    "form_diff": [{ "dimension": "emotion", "before": "", "after": "", "change": "same" }],
    "trend": [{ "label": "基线", "date": "ISO", "conviction": 0 }],
    "narrative": "",
    "suggestion": ""
  }
}
```

复诊对话中，问句以 `role:"assistant"`、用户答以 `role:"user"` 记录，便于复用诊断/确信度的文本格式化。

## 9. API（蓝图 `/api/tracker`）

| 方法 路径 | 请求 | 响应 |
|---|---|---|
| `GET /baselines` | — | `{baselines:[{session_id,title,last_active,turn_count,emotion,cognitive_distortion}]}`（仅 `turn_count≥2` 或有 `automatic_thought` 的会话） |
| `POST /start` | `{baseline_id}` | `{checkin_id, baseline_form, question, q_index, total}`；基线不存在→404 |
| `POST /message` | `{checkin_id, message}` | 进行中：`{done:false, question, q_index, total}`；结束：`{done:true, report}`；未知→404 |
| `GET /checkins?baseline_id=` | — | `{checkins:[{checkin_id,status,created_at,current_conviction,status_label}]}` |
| `GET /report/<checkin_id>` | — | `{report}`；未完成→409；未知→404 |

蓝图在 `webapp/app.py` 注册（与 `chat_bp` 并列）。

## 10. Prompt 草案（供评审）

### 10.1 复诊问句（模板兜底 + 受限润色）
模板见 §6，是唯一信源。润色 prompt（`THERAPIST_*` 模型）：
```
你是一名 CBT 随访助手。下面是一条已基于来访者基线认知评估表生成的"复诊问句模板"。
你的唯一任务：在【不改变其含义、不新增任何话题或建议、不删除其引用的基线内容】的前提下，
把它润色得更自然、温和、口语化。
- 必须原样保留问句中引用的基线片段（情境/自动思维原文）与 0–100 评分要求（若模板含）。
- 不得加入模板之外的新问题、解释或建议。
只输出润色后的一句话，不要引号、不要任何解释。
```
**校验（不过则回退原模板）**：
1. 输出非空且长度 ≤ 模板长度 × 1.8；
2. 模板若含某基线字段原文（如 `automatic_thought` / `situation` 文本），润色结果必须仍包含该子串；
3. 模板若含"0" 与 "100"（评分要求），润色结果必须仍含 "0" 与 "100"。
任一校验失败 → 使用原模板（保证零自创、零漂移）。

### 10.2 对比叙述（judge 模型）
System（**复用**确信度量表语义，不发明）：
```
你是一名临床CBT督导，正在对比来访者前后两次的状态变化。

信念确信度量表（0–100）：100=深信不疑/完全封闭；75=坚定但偶有犹豫；
50=有所动摇；25=明显开放；0=已放弃该负面信念或未体现。

你将收到：基线与当前的 CBT 诊断表、两次的确信度分数及其证据引用、以及来访者本次复诊的回答。
请仅基于这些已提供的信息，客观对比每个维度的变化，引用证据，并：
1. 判断整体状态：改善 / 持平 / 恶化（以确信度变化为主，诊断表变化为辅）。
2. 给出一条温和的、基于CBT原则的下一步建议（如行为实验、认知重构、苏格拉底自问）。
禁止编造任何未提供的信息。

严格输出 JSON：{"status":"改善|持平|恶化","narrative":"...","suggestion":"..."}
```

## 11. 前端「追踪」视图

- 导航栏新增「追踪」按钮 → 切换到 tracker 视图（隐藏 `.layout` 聊天布局，显示 `#tracker-view`）。
- 视图三态：
  1. **基线选择**：列出可追踪会话（`GET /baselines`）。
  2. **复诊进行**：选定后点「开始复诊」→ 逐题问答（复用 `appendUserMsg/appendAiMsg` 气泡）。
  3. **报告**：逐维度 diff 表 + 确信度 SVG 折线趋势图 + 状态徽标 + 叙述 + 建议；可从复诊列表回看。
- 趋势图：无依赖内联 SVG `renderTrendChart(points)`（折线 + 点 + 轴标），与现有暖色风格一致；下降标注为"好转"。
- 复用现有 toast / 气泡 / 卡片样式，保持无缝。

## 12. 错误处理
- 基线/复诊不存在 → 404；报告未就绪 → 409。
- `get_belief_conviction` 失败 → 内部回退 `(50,"解析失败")`（既有行为），报告标注数据不确定。
- 持久化 I/O 失败 → 仅记日志，不中断（沿用 `SessionStore` 既有策略）。
- 复诊中断（用户离开）→ 记录留存 `in_progress`，可从列表续答。

## 13. 测试

### 单元（mock LLM，不耗真实调用）
- `TrackerNode.build_questions`：按非空字段数生成正确问句、含基线插值、空表降级。
- 问句润色校验：保留基线子串/评分要求的润色通过；丢失基线片段或超长的润色被拒并回退原模板；`ENABLE_TRACKER_POLISH=false` 时纯模板。
- 确信度对比与 `conviction_delta` 计算、状态阈值（±15 边界）。
- `form_diff` 各 change 类型（same/changed/new/cleared）。
- 趋势聚合：多复诊按时间排序、点数正确。
- 叙述 JSON 解析与回退。

### 端到端（真实服务 + 真实模型，少量调用）
- 浏览器：进入追踪页 → 选基线 b0af3bf7 → 走完 3–4 题复诊 → 报告出现，含 diff 表、趋势图（≥2 点）、状态、叙述。
- 路由：`/baselines`、`/start`、`/message`×N、`/report` 正常码；未知 404、未完成 409。
- 数据隔离：复诊写入 `results/trackers/`，不污染 `results/sessions/`。

## 14. 实现阶段
1. `agents/tracker.py`（TrackerNode）+ 单测。
2. `webapp/core/tracker_service.py`（编排 + 报告计算，复用 eval/diagnostician）+ 单测。
3. `webapp/routes/tracker.py` + 注册 + 路由测。
4. 前端追踪视图 + SVG 趋势图。
5. 端到端浏览器验证。

## 15. 风险 / 待定
- 确信度对单条文本的稳定性：同一文本多次打分可能有波动（temperature=0 已缓解）。趋势看相对变化而非绝对值。
- LLM 润色漂移：受 §10.1 三项校验 + 模板回退约束；最坏情况退化为纯模板，不影响信源接地。
- 基线若 `automatic_thought` 为空，复诊问句会偏少；§6 第 4 问始终存在以保底。
- 趋势仅 1 次复诊时为 2 点（基线+本次），仍可成图。
