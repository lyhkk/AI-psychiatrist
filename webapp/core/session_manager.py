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
import os
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
from .persistence import SessionStore

logger = logging.getLogger("cbt.webapp.session")


def _build_store() -> SessionStore:
    """根据 .env 配置构建会话持久化存储。"""
    enabled = os.getenv("ENABLE_SESSION_PERSISTENCE", "true").strip().lower() in (
        "1", "true", "yes", "on",
    )
    directory = os.getenv("SESSION_STORE_DIR") or str(
        _PROJECT_ROOT / "results" / "sessions"
    )
    logger.info(
        "[SessionManager] Persistence enabled=%s  dir=%s", enabled, directory
    )
    return SessionStore(directory, enabled=enabled)

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

    def __init__(
        self,
        session_id: str,
        store: SessionStore | None = None,
        *,
        state: DialogueState | None = None,
        created_at: str | None = None,
        turn_count: int = 0,
    ):
        self.session_id: str = session_id
        self.created_at: str = created_at or datetime.now().isoformat()
        self.last_active: str = self.created_at
        self.turn_count: int = turn_count
        # 逐轮诊断表快照：[{"turn": int, "cbt_form": {...}}]
        # 撤销上一轮、Markdown 诊断演变、以及后续 Tracker 状态对比都复用它
        self.cbt_form_history: list[dict[str, Any]] = []
        self._safety = SafetyService()
        self._store = store
        self._io_lock = threading.Lock()

        # LangGraph 编译图（每个 session 独立，避免状态污染）
        self._graph = build_intervention_graph()

        # LangGraph 全局对话状态（空历史，由第一条 chat() 填充；或从磁盘恢复）
        self._state: DialogueState = state if state is not None else make_initial_state()

    # ── 持久化 ──────────────────────────────────────────────────────────────

    def _to_record(self) -> dict[str, Any]:
        """将当前会话序列化为可落盘的纯 dict。"""
        return {
            "session_id":   self.session_id,
            "created_at":   self.created_at,
            "last_active":  self.last_active,
            "turn_count":   self.turn_count,
            "model":        os.getenv("THERAPIST_MODEL") or os.getenv("LLM_MODEL") or "",
            "cbt_form":     dict(self._state.get("cbt_form", {})),
            "cbt_form_history": [dict(s) for s in self.cbt_form_history],
            "chat_history": list(self._state.get("chat_history", [])),
            "entropy_scores": list(self._state.get("entropy_scores", [])),
            "last_inner_monologue":   self._state.get("last_inner_monologue", ""),
            "last_therapist_response": self._state.get("last_therapist_response", ""),
        }

    def _save(self) -> None:
        """将当前状态原子写入磁盘（失败不影响对话）。"""
        if self._store is None:
            return
        with self._io_lock:
            self._store.save(self._to_record())

    @classmethod
    def restore(cls, record: dict[str, Any], store: SessionStore | None) -> "_Session":
        """从磁盘记录重建一个会话实例（用于服务重启后恢复）。"""
        state = make_initial_state()
        state["chat_history"]   = list(record.get("chat_history", []))
        form = record.get("cbt_form") or {}
        if form:
            state["cbt_form"] = dict(form)
        state["entropy_scores"] = list(record.get("entropy_scores", []))
        state["last_inner_monologue"]    = record.get("last_inner_monologue", "")
        state["last_therapist_response"] = record.get("last_therapist_response", "")
        turn_count = int(record.get("turn_count", 0))
        state["turn_count"] = turn_count
        # last_patient_msg 取历史中最后一条用户消息，供下一轮节点参考
        last_user = next(
            (m.get("content", "") for m in reversed(state["chat_history"])
             if m.get("role") == "user"),
            "",
        )
        state["last_patient_msg"] = last_user
        sess = cls(
            record["session_id"],
            store,
            state=state,
            created_at=record.get("created_at"),
            turn_count=turn_count,
        )
        sess.last_active = record.get("last_active") or sess.last_active
        sess.cbt_form_history = [dict(s) for s in record.get("cbt_form_history", [])]
        return sess

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
            # 危机中断轮：诊断表未变，仍记录一份快照使其与轮次对齐
            self.cbt_form_history.append({
                "turn": self.turn_count,
                "cbt_form": dict(self._state.get("cbt_form", {})),
            })
            self._save()

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
        self.cbt_form_history.append({"turn": self.turn_count, "cbt_form": cbt_form})

        logger.info(
            "[Session:%s] turn=%d  distortion=%s  emotion=%s",
            self.session_id[:8],
            self.turn_count,
            cbt_form.get("cognitive_distortion"),
            cbt_form.get("emotion"),
        )

        self._save()

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

    def undo_last_turn(self) -> dict[str, Any]:
        """
        撤销最近一轮：移除最后一对 user/assistant 消息、轮次 -1、
        并把诊断表回退到上一轮快照。

        Raises
        ------
        ValueError : 当前没有可撤销的轮次
        """
        if self.turn_count <= 0:
            raise ValueError("当前没有可撤销的轮次")

        history = list(self._state.get("chat_history", []))
        # 一轮 = 末尾的 user + assistant 两条；按角色稳妥剥离
        if history and history[-1].get("role") == "assistant":
            history.pop()
        if history and history[-1].get("role") == "user":
            history.pop()
        self._state["chat_history"] = history

        # 弹出本轮诊断快照，诊断表回退到上一轮（无则回到空表）
        if self.cbt_form_history:
            self.cbt_form_history.pop()
        if self.cbt_form_history:
            prev_form = dict(self.cbt_form_history[-1].get("cbt_form", {}))
        else:
            prev_form = make_initial_state()["cbt_form"]
        self._state["cbt_form"] = prev_form

        self.turn_count = max(0, self.turn_count - 1)
        self._state["turn_count"] = self.turn_count
        self._state["last_patient_msg"] = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "user"),
            "",
        )
        self._state["last_therapist_response"] = next(
            (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
            "",
        )
        self.last_active = datetime.now().isoformat()
        self._save()
        return {
            "session_id": self.session_id,
            "turn":       self.turn_count,
            "history":    history,
            "cbt_form":   dict(prev_form),
        }

    def summary(self) -> dict[str, Any]:
        """生成用于历史列表的轻量摘要。"""
        history = self._state.get("chat_history", [])
        first_user = next(
            (m.get("content", "") for m in history if m.get("role") == "user"), ""
        )
        title = (first_user[:40] + "…") if len(first_user) > 40 else (first_user or "(空会话)")
        form = self._state.get("cbt_form", {}) or {}
        return {
            "session_id":  self.session_id,
            "created_at":  self.created_at,
            "last_active": self.last_active,
            "turn_count":  self.turn_count,
            "title":       title,
            "emotion":     form.get("emotion"),
            "cognitive_distortion": form.get("cognitive_distortion"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 摘要工具（供磁盘记录直接生成列表项，无需实例化 _Session）
# ─────────────────────────────────────────────────────────────────────────────

def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    """从磁盘记录直接抽取历史列表摘要（容错处理异常记录）。"""
    history = record.get("chat_history", []) or []
    first_user = next(
        (m.get("content", "") for m in history
         if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )
    title = (first_user[:40] + "…") if len(first_user) > 40 else (first_user or "(空会话)")
    form = record.get("cbt_form", {}) or {}
    return {
        "session_id":  record.get("session_id", ""),
        "created_at":  record.get("created_at", ""),
        "last_active": record.get("last_active", ""),
        "turn_count":  int(record.get("turn_count", 0) or 0),
        "title":       title,
        "emotion":     form.get("emotion"),
        "cognitive_distortion": form.get("cognitive_distortion"),
    }


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
        self._store = _build_store()

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
        sess = _Session(session_id, self._store)
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

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        列出全部会话摘要（内存中的实时状态优先，其余取自磁盘存档），
        按最后活跃时间倒序。
        """
        summaries: dict[str, dict[str, Any]] = {}
        # 磁盘存档
        if self._store is not None:
            for record in self._store.list_records():
                sid = record.get("session_id")
                if sid:
                    summaries[sid] = _record_summary(record)
        # 内存中的活跃会话覆盖磁盘版本（状态更新）
        with self._lock:
            for sid, sess in self._sessions.items():
                summaries[sid] = sess.summary()
        return sorted(
            summaries.values(),
            key=lambda s: s.get("last_active") or "",
            reverse=True,
        )

    def get_record(self, session_id: str) -> dict[str, Any]:
        """
        获取一条完整会话记录（含 chat_history / cbt_form / cbt_form_history）。
        活跃会话取内存实时快照，否则读磁盘存档。

        Raises
        ------
        KeyError : 会话既不在内存也不在磁盘
        """
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is not None:
            return sess._to_record()
        if self._store is not None:
            record = self._store.load(session_id)
            if record is not None:
                return record
        raise KeyError(f"Session not found: {session_id}")

    def undo_last_turn(self, session_id: str) -> dict[str, Any]:
        """撤销指定会话的最近一轮。"""
        return self._get_session(session_id).undo_last_turn()

    def close_session(self, session_id: str) -> None:
        """销毁指定会话，释放内存（磁盘存档保留，便于后续查阅/恢复）。"""
        with self._lock:
            self._sessions.pop(session_id, None)
        logger.info("[SessionManager] Closed session %s (disk archive kept)", session_id[:8])

    def active_count(self) -> int:
        """当前活跃会话数。"""
        return len(self._sessions)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> _Session:
        with self._lock:
            session = self._sessions.get(session_id)
            # 内存未命中：尝试从磁盘恢复（服务重启后仍可续聊）
            if session is None and self._store is not None:
                record = self._store.load(session_id)
                if record is not None:
                    session = _Session.restore(record, self._store)
                    self._sessions[session_id] = session
                    logger.info(
                        "[SessionManager] Rehydrated session %s from disk (turn=%d)",
                        session_id[:8], session.turn_count,
                    )
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

