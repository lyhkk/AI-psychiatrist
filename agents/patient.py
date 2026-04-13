"""
patient.py — 患者模拟器（Patient Agent）

模型配置：优先 PATIENT_* env 变量，回退到 SUPERVISOR_*。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .llm_base import LLMClient
from .state import DialogueState

logger = logging.getLogger("cbt.patient")

_DEFAULT_BACKGROUND = (
    "你是一名大三学生，因连续两次面试失败而感到极度绝望。"
    "你认为自己永远找不到工作，是个彻底的失败者。"
)


def _build_system_prompt(background: str) -> str:
    """
    根据结构化背景剧本构建患者角色 system prompt。

    background 由 run_simulation.py 按以下结构拼装：
      【来访者主诉标题】...
      【详细描述（来访者原始陈述）】...
      【话题标签】...
      【参考咨询师视角（仅用于推断来访者深层情绪，不要直接引用）】...
    """
    return (
        "你正在扮演一名寻求心理咨询的来访者。请严格根据以下背景资料塑造角色，\n"
        "确保角色的语言、情绪反应和防御模式与资料中描述的情况高度一致。\n\n"
        "══════════════════════════════════════\n"
        "【角色背景资料（严格按此扮演）】\n"
        "══════════════════════════════════════\n"
        + background + "\n\n"
        "══════════════════════════════════════\n"
        "【角色扮演规则】\n"
        "══════════════════════════════════════\n"
        "1. 【情绪真实性】严格基于上方描述的具体情境和感受作出反应，\n"
        "   不要使用通用的焦虑或抑郁表达，要引用背景中的具体细节。\n"
        "2. 【高防御型人格】你充满防御心，固执地维护自己的认知框架。\n"
        "   - 若咨询师直接给建议、急于否定你的看法，你会感到被冒犯并抵触。\n"
        "   - 若咨询师无底线迎合你的负面情绪，你会感到厌烦，认为对方在敷衍。\n"
        "   - 只有咨询师展现深刻共情并用苏格拉底式问题引导时，你才会逐渐开放。\n"
        "3. 【渐进式松动】即便内心有所触动，也只是稍微松动，绝不突然转变立场。\n"
        "   可以用'……也许吧'、'道理是这样，但是……'等方式表现矛盾心理。\n"
        "4. 【语言风格】用日常口语，语气符合真实来访者状态，每次回复不超过100字。\n"
        "5. 【角色边界】你只是来访者，不要扮演咨询师、不要给出建议、不要跳出角色。\n"
    )


class PatientNode:
    """
    LangGraph 节点：患者模拟器。
    优先读取 PATIENT_* env 变量，回退到 SUPERVISOR_*。
    沙盘模拟中由 run_simulation.py 驱动；真实系统中不启用。
    """

    def __init__(self, llm_client: LLMClient | None = None, background: str = ""):
        if llm_client is not None:
            self.llm = llm_client
        else:
            self.llm = LLMClient(
                api_key=os.getenv("PATIENT_API_KEY") or os.getenv("SUPERVISOR_API_KEY"),
                base_url=os.getenv("PATIENT_BASE_URL") or os.getenv("SUPERVISOR_BASE_URL"),
                model=os.getenv("PATIENT_MODEL") or os.getenv("SUPERVISOR_MODEL"),
                role_label="patient",
            )
        self.background = background or _DEFAULT_BACKGROUND
        self._system_prompt = _build_system_prompt(self.background)
        self._history: list[dict[str, str]] = []

    def reset(self) -> None:
        """重置内部对话历史（开始新会话时调用）。"""
        self._history = []

    def set_background(
        self,
        question: str,
        description: str = "",
        keywords: str = "",
        answers: list | None = None,
    ) -> None:
        """
        从 PsyQA 条目更新患者背景剧本，注入完整结构化信息。

        Parameters
        ----------
        question    : 来访者主诉标题
        description : 来访者详细描述（原始陈述）
        keywords    : 话题标签
        answers     : 咨询师参考回答列表（取第一条用于推断深层情绪）
        """
        parts = []
        if question:
            parts.append("【来访者主诉标题】\n" + question)
        if description:
            parts.append("【详细描述（来访者原始陈述）】\n" + description)
        if keywords:
            parts.append("【话题标签】" + keywords)
        if answers:
            first = answers[0]
            answer_text = first.get("answer_text", "") if isinstance(first, dict) else str(first)
            if answer_text:
                parts.append(
                    "【参考咨询师视角（仅用于推断来访者深层情绪，不要直接引用）】\n"
                    + answer_text[:500]
                )
        self.background = "\n\n".join(parts) if parts else question
        self._system_prompt = _build_system_prompt(self.background)
        logger.info("[Patient] Background updated: %s | keywords=%s", question[:60], keywords)

    def __call__(self, state: DialogueState) -> dict[str, Any]:
        """接收治疗师最新回复，生成患者下一句话，更新全局 state。"""
        therapist_reply = state.get("last_therapist_response", "")
        if not therapist_reply:
            logger.warning("[Patient] No therapist reply found in state")
            return {}

        # 患者侧视角：治疗师说的是 user，患者自己是 assistant
        self._history.append({"role": "user", "content": therapist_reply})
        messages = [{"role": "system", "content": self._system_prompt}] + self._history

        try:
            patient_reply = self.llm.chat(messages=messages, temperature=0.85)
        except Exception as exc:
            logger.error("[Patient] LLM call failed: %s", exc)
            patient_reply = "我不知道该说什么。"

        self._history.append({"role": "assistant", "content": patient_reply})
        logger.info("[Patient] turn=%d | reply: %s",
                    state.get("turn_count", 0), patient_reply[:80])
        logger.debug("[Patient] turn=%d | FULL reply:\n%s",
                     state.get("turn_count", 0), patient_reply)

        updated_history = list(state.get("chat_history", [])) + [
            {"role": "user", "content": patient_reply}
        ]
        return {"last_patient_msg": patient_reply, "chat_history": updated_history}
