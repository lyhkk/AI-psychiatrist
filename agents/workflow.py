"""
workflow.py
LangGraph 干预工作流：DiagnosticianNode -> TherapistNode。

模型配置完全从 .env 读取，无需手动传参：
  - DiagnosticianNode 使用 SUPERVISOR_* 配置
  - TherapistNode     使用 THERAPIST_*  配置
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from .diagnostician import DiagnosticianNode
from .llm_base import LLMClient
from .state import DialogueState
from .therapist import TherapistNode

logger = logging.getLogger("cbt.workflow")

NODE_DIAGNOSTICIAN = "diagnostician"
NODE_THERAPIST = "therapist"


def build_intervention_graph(
    llm_client: LLMClient | None = None,
    diagnostician_llm: LLMClient | None = None,
    therapist_llm: LLMClient | None = None,
) -> Any:
    """
    构建并编译 LangGraph 干预工作流图。

    图结构: [入口] -> diagnostician -> therapist -> [END]

    参数说明（均可不传，自动从 .env 读取对应角色配置）：
    - llm_client       : 若指定，两个节点共享同一客户端（用于简单测试）
    - diagnostician_llm: 为 Diagnostician 单独指定客户端（覆盖 SUPERVISOR_*）
    - therapist_llm    : 为 Therapist 单独指定客户端（覆盖 THERAPIST_*）
    """
    d_llm = diagnostician_llm or llm_client or None
    t_llm = therapist_llm or llm_client or None

    diagnostician = DiagnosticianNode(llm_client=d_llm)
    therapist = TherapistNode(llm_client=t_llm)

    graph = StateGraph(DialogueState)
    graph.add_node(NODE_DIAGNOSTICIAN, diagnostician)
    graph.add_node(NODE_THERAPIST, therapist)
    graph.set_entry_point(NODE_DIAGNOSTICIAN)
    graph.add_edge(NODE_DIAGNOSTICIAN, NODE_THERAPIST)
    graph.add_edge(NODE_THERAPIST, END)

    compiled = graph.compile()
    logger.info(
        "[Workflow] Graph compiled: %s -> %s -> END  (diag_model=%s, therapist_model=%s)",
        NODE_DIAGNOSTICIAN, NODE_THERAPIST,
        diagnostician.llm.model, therapist.llm.model,
    )
    return compiled


def run_single_turn(compiled_graph: Any, state: DialogueState) -> DialogueState:
    """Execute one full intervention turn (Diagnostician + Therapist)."""
    return compiled_graph.invoke(state)
