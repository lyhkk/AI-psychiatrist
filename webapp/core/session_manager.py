"""
webapp/core/session_manager.py

会话管理层 —— 将 agents 系统（LangGraph 干预图）封装为面向 Flask 的服务。

设计原则
--------
- 与 Flask 完全解耦：此模块不 import 任何 Flask 对象，可单独测试。
- 与 agents 完全解耦：Flask 路由层只调用此模块，不直接接触 LangGraph。
- 每个浏览器 session_id 持有一个独立的 CBT 对话状态（DialogueState）。
- SessionManager 作为单例由 Flask app 在启动时创建，通过 get_session_manager() 获取。

升级 agents 系统时：只需保证 agents 包的公共接口不变（build_intervention_graph /
make_initial_state / DialogueState 字段），本模块无需改动。
"""

from __future__ import annotations

import logging
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保项目根目录在 Python 路径中，使 agents 包可被正确导入
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from agents import (
    build_intervention_graph,
    make_initial_state,
    DialogueState,
)
from .safety import SafetyService

logger = logging.getLogger("cbt.webapp.session")

# ─────────────────────────────────────────────────────────────────────────────
# 单次对话轮次的返回结构（纯 Python dict，方便序列化为 JSON）
# ─────────────────────────────────────────────────────────────────────────────

TurnResult = dict[str, Any]
"""
{
  "session_id"    : str,
  "turn"          : int,
  "therapist_reply": str,
  "cbt_form"      : dict,   # 当前认知评估表快照
  "timestamp"     : str,    # ISO 8601
  "is_first_turn" : bool,
  "interrupted"   : bool,
  "safety_category": str | None,
}
"""


class _Session:
    """单个用户会话的内部状态容器。"""

    def __init__(self, session_id: str):
        self.session_id: str = session_id
        self.created_at: str = datetime.now().isoformat()
        self.last_active: str = self.created_at
        self.turn_count: int = 0
        self._safety = SafetyService()

        # LangGraph 编译图（每个 session 独立，避免状态污染）
        self._graph = build_intervention_graph()

        # LangGraph 全局对话状态（空历史，由第一条 chat() 填充）
        self._state: DialogueState = make_initial_state()

    def advance(
        self, user_message: str
    ) -> TurnResult:
        """
        向治疗师发送一条用户消息，推进一个对话轮次。

        Parameters
        ----------
        user_message : 用户（来访者）本轮输入文本

        Returns
        -------
        TurnResult dict
        """
        user_safety = self._safety.check_user_message(user_message)
        if user_safety.flagged:
            history = list(self._state.get("chat_history", []))
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": user_safety.message or ""})
            self._state["chat_history"] = history
            self._state["last_patient_msg"] = user_message
            self._state["last_therapist_response"] = user_safety.message or ""
            self.turn_count += 1
            self._state["turn_count"] = self.turn_count
            self.last_active = datetime.now().isoformat()

            return {
                "session_id": self.session_id,
                "turn": self.turn_count,
                "therapist_reply": user_safety.message or "",
                "cbt_form": dict(self._state.get("cbt_form", {})),
                "timestamp": self.last_active,
                "is_first_turn": self.turn_count == 1,
                "interrupted": True,
                "safety_category": user_safety.category,
            }

        # 将用户消息追加到 chat_history
        history = list(self._state.get("chat_history", []))
        history.append({"role": "user", "content": user_message})
        self._state["chat_history"] = history
        self._state["last_patient_msg"] = user_message

        # 调用 LangGraph 图（Diagnostician → Therapist）
        self._state = self._graph.invoke(self._state)

        therapist_reply = self._state.get("last_therapist_response", "")
        output_safety = self._safety.check_model_output(therapist_reply)
        if output_safety.flagged:
            safe_reply = output_safety.message or ""
            self._state["last_therapist_response"] = safe_reply
            updated_history = list(self._state.get("chat_history", []))
            if updated_history and updated_history[-1].get("role") == "assistant":
                updated_history[-1] = {"role": "assistant", "content": safe_reply}
            else:
                updated_history.append({"role": "assistant", "content": safe_reply})
            self._state["chat_history"] = updated_history
            therapist_reply = safe_reply

        self.turn_count += 1
        self._state["turn_count"] = self.turn_count
        self.last_active = datetime.now().isoformat()

        cbt_form = dict(self._state.get("cbt_form", {}))

        logger.info(
            "[Session:%s] turn=%d  distortion=%s  emotion=%s",
            self.session_id[:8],
            self.turn_count,
            cbt_form.get("cognitive_distortion"),
            cbt_form.get("emotion"),
        )

        return {
            "session_id":     self.session_id,
            "turn":           self.turn_count,
            "therapist_reply": therapist_reply,
            "cbt_form":       cbt_form,
            "timestamp":      self.last_active,
            "is_first_turn":  self.turn_count == 1,
            "interrupted":    False,
            "safety_category": output_safety.category if output_safety.flagged else None,
        }

    def get_history(self) -> list[dict[str, str]]:
        """返回当前完整对话历史（原始 chat_history 列表）。"""
        return list(self._state.get("chat_history", []))

    def get_cbt_form(self) -> dict:
        """返回当前认知评估表快照。"""
        return dict(self._state.get("cbt_form", {}))


# ─────────────────────────────────────────────────────────────────────────────
# SessionManager 单例
# ─────────────────────────────────────────────────────────────────────────────

class SessionManager:
    """
    管理所有用户会话的生命周期。

    - create_session()  : 新建会话，返回 session_id
    - chat()            : 推进一个对话轮次
    - get_history()     : 获取对话历史
    - get_cbt_form()    : 获取认知评估表
    - close_session()   : 主动销毁会话
    """

    def __init__(self):
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    # ── 公共接口 ──────────────────────────────────────────────────────────────

    def create_session(self) -> str:
        """
        创建新的 CBT 对话会话。
        开场白由调用方随后通过 chat() 发送，不在此处写入 state。

        Returns
        -------
        session_id : str  （UUID4 字符串）
        """
        session_id = str(uuid.uuid4())
        sess = _Session(session_id)
        with self._lock:
            self._sessions[session_id] = sess
        logger.info("[SessionManager] Created session %s", session_id[:8])
        return session_id

    def chat(
        self, session_id: str, user_message: str
    ) -> TurnResult:
        """
        向指定会话发送用户消息，返回治疗师回复。

        Raises
        ------
        KeyError   : session_id 不存在
        RuntimeError : LangGraph 执行失败
        """
        session = self._get_session(session_id)
        try:
            return session.advance(user_message)
        except Exception as exc:
            logger.error(
                "[SessionManager] Session %s advance failed: %s",
                session_id[:8], exc,
            )
            raise RuntimeError(f"对话推进失败：{exc}") from exc

    def get_history(
        self, session_id: str
    ) -> list[dict[str, str]]:
        """获取指定会话的完整对话历史。"""
        return self._get_session(session_id).get_history()

    def get_cbt_form(self, session_id: str) -> dict:
        """获取指定会话的最新认知评估表。"""
        return self._get_session(session_id).get_cbt_form()

    def close_session(self, session_id: str) -> None:
        """销毁指定会话，释放内存。"""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info("[SessionManager] Closed session %s", session_id[:8])

    def active_count(self) -> int:
        """当前活跃会话数。"""
        return len(self._sessions)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session


# ─────────────────────────────────────────────────────────────────────────────
# 应用级单例（由 Flask app 在启动时初始化）
# ─────────────────────────────────────────────────────────────────────────────

_manager_instance: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """
    返回全局 SessionManager 单例。
    首次调用时自动创建实例。
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = SessionManager()
        logger.info("[SessionManager] Singleton initialized")
    return _manager_instance

