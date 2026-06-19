"""
webapp/core/export.py

会话导出渲染层 —— 将存储记录（dict）渲染为人类可读的 Markdown。

设计原则
--------
- 纯函数、无副作用、仅依赖标准库；输入是 persistence 记录 dict。
- JSON 导出直接用记录本身，无需本模块；本模块只负责 Markdown 排版。
- 对缺失字段全部容错（用占位符），保证任何记录都能导出。
"""

from __future__ import annotations

from typing import Any

_ROLE_LABEL = {"user": "来访者", "assistant": "咨询师"}
_PLACEHOLDER = "（未识别）"


def _cell(value: Any) -> str:
    """表格单元格安全渲染：None/空 → 占位符，转义竖线与换行。"""
    if value is None or value == "":
        return _PLACEHOLDER
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(record: dict[str, Any]) -> str:
    """将一条会话记录渲染为 Markdown 文本。"""
    sid = record.get("session_id", "unknown")
    form = record.get("cbt_form", {}) or {}
    lines: list[str] = []

    # ── 标题与元信息 ──────────────────────────────────────────────
    lines.append(f"# CBT 会话记录 · {sid[:8]}")
    lines.append("")
    lines.append(f"- 会话 ID：`{sid}`")
    lines.append(f"- 创建时间：{record.get('created_at') or _PLACEHOLDER}")
    lines.append(f"- 最后活跃：{record.get('last_active') or _PLACEHOLDER}")
    lines.append(f"- 对话轮次：{record.get('turn_count', 0)}")
    lines.append(f"- 模型：{record.get('model') or _PLACEHOLDER}")
    lines.append("")

    # ── 认知评估诊断结果 ──────────────────────────────────────────
    lines.append("## 认知评估诊断结果")
    lines.append("")
    lines.append("| 维度 | 内容 |")
    lines.append("| --- | --- |")
    lines.append(f"| 情境 | {_cell(form.get('situation'))} |")
    lines.append(f"| 情绪 | {_cell(form.get('emotion'))} |")
    lines.append(f"| 自动思维 | {_cell(form.get('automatic_thought'))} |")
    lines.append(f"| 认知扭曲 | {_cell(form.get('cognitive_distortion'))} |")
    lines.append("")

    # ── 诊断演变（逐轮快照）──────────────────────────────────────
    history = record.get("cbt_form_history", []) or []
    if history:
        lines.append("## 诊断演变")
        lines.append("")
        lines.append("| 轮次 | 情绪 | 认知扭曲 |")
        lines.append("| --- | --- | --- |")
        for snap in history:
            turn = snap.get("turn", "")
            sform = snap.get("cbt_form", {}) or {}
            lines.append(
                f"| {turn} | {_cell(sform.get('emotion'))} | "
                f"{_cell(sform.get('cognitive_distortion'))} |"
            )
        lines.append("")

    # ── 对话记录 ──────────────────────────────────────────────────
    lines.append("## 对话记录")
    lines.append("")
    chat = record.get("chat_history", []) or []
    if not chat:
        lines.append("_（无对话内容）_")
    else:
        for msg in chat:
            role = _ROLE_LABEL.get(msg.get("role", ""), msg.get("role", "?"))
            content = (msg.get("content") or "").strip()
            lines.append(f"**{role}：** {content}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
