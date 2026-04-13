"""
therapist.py
MDP-CoT 治疗师（Therapist Agent）。

职责：在 Diagnostician 更新表单后运行，生成对患者的回复。
必须先在 <inner_monologue> 中进行记忆驱动动态规划，再在 <response> 中输出对话。
输出经过 XML 标签解析，严格分离 inner_monologue 和 response。

模型配置：读取 .env 中的 THERAPIST_* 变量（前端对话模型）。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .llm_base import LLMClient
from .state import DialogueState

logger = logging.getLogger("cbt.therapist")


_SYSTEM_TEMPLATE = """\
你是一位专业的CBT心理咨询师。

【后台认知评估表（仅供你内部参考，禁止直接暴露给患者）】
- 情境 (situation)            : {situation}
- 情绪 (emotion)              : {emotion}
- 自动思维 (automatic_thought) : {automatic_thought}
- 认知扭曲类别 (cognitive_distortion) : {cognitive_distortion}

在回复患者之前，你【必须】首先在 <inner_monologue> 标签内进行思考（此内容患者不可见）：
1. **防御评估**：患者当前的情绪张力和防御水平如何？是否有抵触迹象？
2. **表单缺口分析**：目前还有哪些字段是 null 需要通过苏格拉底提问探询？
   或者表单是否已足够完整，可以发起认知面质？
3. **策略规划**：根据以上判断，决定本轮的话术方向：
   共情陪伴 / 开放式探询 / 引导式发现 / 认知重构 / 行为激活 等。

思考完成后，在 <response> 标签内输出自然、温和的中文回复。

【硬性规则】
- 切勿在 <response> 中暴露后台表单任何内容。
- 切勿在 <response> 中直接给出说教式建议。
- 每次回复尽量聚焦，不超过 150 字。
- 输出格式必须严格遵守（不要额外文字）：

<inner_monologue>
[你的内心独白]
</inner_monologue>
<response>
[你对患者的自然回复]
</response>
"""

# 每轮强制追加在消息末尾的格式提醒（防止长对话中模型遗忘格式约束）
_FORMAT_REMINDER = (
    "【格式强制要求】请严格按以下结构输出，不得省略任何标签：\n"
    "<inner_monologue>\n[防御评估 / 表单缺口分析 / 策略规划]\n</inner_monologue>\n"
    "<response>\n[对患者的自然中文回复，不超过150字]\n</response>"
)


def _parse_xml_output(raw: str) -> tuple[str, str]:
    """
    从模型输出中解析 <inner_monologue> 和 <response> 两段内容。
    Returns: (inner_monologue, response)
    """
    mono_match = re.search(
        r"<inner_monologue>\s*([\s\S]*?)\s*</inner_monologue>",
        raw, re.IGNORECASE
    )
    resp_match = re.search(
        r"<response>\s*([\s\S]*?)\s*</response>",
        raw, re.IGNORECASE
    )
    inner_monologue = mono_match.group(1).strip() if mono_match else ""
    response = resp_match.group(1).strip() if resp_match else raw.strip()

    if not mono_match:
        logger.warning("[Therapist] <inner_monologue> tag not found in output")
    if not resp_match:
        logger.warning("[Therapist] <response> tag not found, using raw output")

    return inner_monologue, response


class TherapistNode:
    """
    LangGraph 节点：MDP-CoT 治疗师。
    默认使用 .env 中 THERAPIST_* 配置的前端对话模型。
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or LLMClient.from_role("therapist")

    def _build_system_prompt(self, cbt_form: dict) -> str:
        return _SYSTEM_TEMPLATE.format(
            situation=cbt_form.get("situation") or "未知",
            emotion=cbt_form.get("emotion") or "未知",
            automatic_thought=cbt_form.get("automatic_thought") or "未知",
            cognitive_distortion=cbt_form.get("cognitive_distortion") or "尚未确定",
        )

    def __call__(self, state: DialogueState) -> dict[str, Any]:
        cbt_form = state.get("cbt_form", {})
        system_prompt = self._build_system_prompt(cbt_form)

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(state.get("chat_history", []))
        # 在每轮末尾追加格式提醒，防止长对话中模型遗忘 XML 结构约束
        messages.append({"role": "user", "content": _FORMAT_REMINDER})

        try:
            raw = self.llm.chat(messages=messages, temperature=0.75)
        except Exception as exc:
            logger.error("[Therapist] LLM call failed: %s", exc)
            raw = "<response>非常抱歉，我现在无法回应，请稍后再试。</response>"

        inner_monologue, response = _parse_xml_output(raw)

        if not inner_monologue:
            logger.warning(
                "[Therapist] turn=%d | <inner_monologue> missing — full raw output:\n%s",
                state.get("turn_count", 0) + 1,
                raw,
            )
        else:
            logger.debug(
                "[Therapist] turn=%d | FULL inner_monologue:\n%s",
                state.get("turn_count", 0) + 1,
                inner_monologue,
            )

        logger.debug(
            "[Therapist] turn=%d | FULL response:\n%s",
            state.get("turn_count", 0) + 1,
            response,
        )
        logger.info(
            "[Therapist] turn=%d | monologue_len=%d | response: %s",
            state.get("turn_count", 0) + 1,
            len(inner_monologue),
            response[:80],
        )

        updated_history = list(state.get("chat_history", [])) + [
            {"role": "assistant", "content": response}
        ]

        return {
            "last_therapist_response": response,
            "last_inner_monologue": inner_monologue,
            "chat_history": updated_history,
            "turn_count": state.get("turn_count", 0) + 1,
        }
