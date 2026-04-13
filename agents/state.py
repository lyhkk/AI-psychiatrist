"""
state.py
LangGraph 全局共享状态定义。
所有智能体节点通过读写 DialogueState 进行交互，不直接传递消息。
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# CBT 认知评估表
# ─────────────────────────────────────────────────────────────────────────────

class CBTForm(TypedDict, total=False):
    """结构化认知行为表单，由 DiagnosticianNode 持续更新。"""
    situation: str | None            # 情境
    emotion: str | None              # 情绪
    automatic_thought: str | None    # 自动思维
    cognitive_distortion: str | None # 认知扭曲类别


# ─────────────────────────────────────────────────────────────────────────────
# 全局对话状态
# ─────────────────────────────────────────────────────────────────────────────

class DialogueState(TypedDict):
    """
    LangGraph StateGraph 使用的全局共享状态。

    字段说明
    --------
    chat_history    : 标准多轮对话消息列表，格式 [{"role": "user"|"assistant", "content": "..."}]
    cbt_form        : 结构化认知行为表单，初始所有字段为 None
    entropy_scores  : 每轮对话的信息熵值记录，用于 IG-PQA 计算
    last_patient_msg: 患者最新一条消息（供节点内部传递使用）
    last_therapist_response: 治疗师最新对外回复
    last_inner_monologue   : 治疗师最新内心独白（<inner_monologue> 内容）
    turn_count      : 当前对话轮数
    """
    chat_history: list[dict[str, str]]
    cbt_form: CBTForm
    entropy_scores: list[float]
    last_patient_msg: str
    last_therapist_response: str
    last_inner_monologue: str
    turn_count: int


def make_initial_state(patient_opening: str = "") -> DialogueState:
    """创建初始状态，所有 cbt_form 字段为 None。"""
    return DialogueState(
        chat_history=(
            [{"role": "user", "content": patient_opening}]
            if patient_opening else []
        ),
        cbt_form=CBTForm(
            situation=None,
            emotion=None,
            automatic_thought=None,
            cognitive_distortion=None,
        ),
        entropy_scores=[],
        last_patient_msg=patient_opening,
        last_therapist_response="",
        last_inner_monologue="",
        turn_count=0,
    )
