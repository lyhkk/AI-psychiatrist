"""
diagnostician.py
临床循证追踪器（Diagnostician / Supervisor Agent）。

职责：在每次患者输入后作为"静默节点"运行，不生成对外对话。
仅从最新对话中提取认知信息，更新 cbt_form 中的空缺字段。
使用 Pydantic 严格校验输出格式。

模型配置：读取 .env 中的 SUPERVISOR_* 变量（后台高推理模型）。
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from .llm_base import LLMClient
from .state import CBTForm, DialogueState

logger = logging.getLogger("cbt.diagnostician")

COGNITIVE_DISTORTIONS = [
    "非此即彼（全或无思维）", "以偏概全", "心理过滤", "否定正面思考",
    "读心术", "先知错误（灾难化）", "放大或缩小", "情绪化推理",
    "应该式", "乱贴标签", "罪责归己", "罪责归他",
]

_SYSTEM_PROMPT = """\
你是一个后台临床循证追踪器。你的唯一任务是阅读最新对话，并更新患者的CBT认知评估表。

提取规则：
1. 从对话中抽取情境（situation）、情绪（emotion）、自动思维（automatic_thought）。
   若发现符合的，填入对应字段；若信息不足，保持为 null。
2. 当收集到足够信息时，严格从以下12种标准认知扭曲中匹配并打上标签：
   非此即彼（全或无思维）、以偏概全、心理过滤、否定正面思考、读心术、
   先知错误（灾难化）、放大或缩小、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他。
   若信息不足，cognitive_distortion 保持为 null。
3. 只更新有新信息的字段，已有内容若无更好证据则保持不变。

严格输出 JSON 对象，不要任何额外文字：
{"situation": "...", "emotion": "...", "automatic_thought": "...", "cognitive_distortion": "..."}
"""


class CBTFormSchema(BaseModel):
    situation: str | None = Field(None)
    emotion: str | None = Field(None)
    automatic_thought: str | None = Field(None)
    cognitive_distortion: str | None = Field(None)


class DiagnosticianNode:
    """
    LangGraph 节点：临床循证追踪器（后台静默节点）。
    默认使用 .env 中 SUPERVISOR_* 配置的高推理后台模型。
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient.from_role("diagnostician")

    def _format_history(self, chat_history: list[dict[str, str]]) -> str:
        lines = []
        for msg in chat_history[-6:]:
            role = "来访者" if msg["role"] == "user" else "咨询师"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    def __call__(self, state: DialogueState) -> dict[str, Any]:
        history_text = self._format_history(state["chat_history"])
        current_form = state.get("cbt_form", {})

        user_prompt = (
            f"当前已知的认知评估表：\n"
            f"  situation: {repr(current_form.get('situation'))}\n"
            f"  emotion: {repr(current_form.get('emotion'))}\n"
            f"  automatic_thought: {repr(current_form.get('automatic_thought'))}\n"
            f"  cognitive_distortion: {repr(current_form.get('cognitive_distortion'))}\n\n"
            f"最新对话记录：\n{history_text}\n\n"
            f"请输出更新后的完整 JSON 评估表。"
        )

        try:
            raw = self.llm.simple_chat(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                temperature=0.0,
            )
            logger.debug(
                "[Diagnostician] SYSTEM PROMPT:\n%s\n\nUSER PROMPT:\n%s\n\nRAW REPLY:\n%s",
                _SYSTEM_PROMPT, user_prompt, raw,
            )
            parsed = self.llm.extract_json(raw)
            if parsed:
                schema = CBTFormSchema(**parsed)
                new_form = CBTForm(
                    situation=schema.situation,
                    emotion=schema.emotion,
                    automatic_thought=schema.automatic_thought,
                    cognitive_distortion=schema.cognitive_distortion,
                )
                logger.info(
                    "[Diagnostician] distortion=%s  emotion=%s",
                    schema.cognitive_distortion, schema.emotion,
                )
                logger.debug(
                    "[Diagnostician] updated cbt_form: situation=%s | automatic_thought=%s",
                    schema.situation, schema.automatic_thought,
                )
                return {"cbt_form": new_form}
            else:
                logger.warning("[Diagnostician] JSON parse failed, keeping existing form")
        except Exception as exc:
            logger.warning("[Diagnostician] LLM call failed: %s", exc)

        return {}
