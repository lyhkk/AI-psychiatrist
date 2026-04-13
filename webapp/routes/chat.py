"""
webapp/routes/chat.py

Flask 蓝图：对话 API（/api/chat/*）。

设计原则
--------
- 此文件只处理 HTTP 请求/响应的编解码，不含任何业务逻辑。
- 所有 AI 调用都委托给 core.session_manager，与 agents 系统完全隔离。
- 升级 agents 系统或修改 Flask 路由时，两侧相互独立。

端点
----
POST /api/chat/start          开始新的 CBT 会话
POST /api/chat/message        发送消息，获取治疗师回复（SSE 流式）
GET  /api/chat/history        获取当前会话的完整对话历史
GET  /api/chat/cbt_form       获取当前认知评估表
DELETE /api/chat/session      结束并销毁当前会话
"""

from __future__ import annotations

import logging
from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    session,
)

from webapp.core import get_session_manager

logger = logging.getLogger("cbt.webapp.routes")

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")


def _ok(data: dict) -> Response:
    return jsonify({"status": "ok", **data})


def _err(message: str, code: int = 400) -> tuple[Response, int]:
    return jsonify({"status": "error", "message": message}), code


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/start
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.post("/start")
def start_session():
    """
    开始新的 CBT 咨询会话。

    Body (JSON, optional)
    ---------------------
    opening : str   来访者开场白（若不传则由前端第一条消息触发）

    Response
    --------
    { status, session_id, message }
    """
    body = request.get_json(silent=True) or {}
    opening = body.get("opening", "").strip()

    mgr = get_session_manager()

    # 若当前浏览器 session 已有会话，先关闭旧的
    old_sid = session.get("session_id")
    if old_sid:
        mgr.close_session(old_sid)

    # opening 仅做日志记录，不写入 state。
    # 开场白由前端随后调用 /message 发送，避免 chat_history 重复追加。
    new_sid = mgr.create_session()
    session["session_id"] = new_sid

    logger.info("[Route] /start  new_session=%s  opening=%s", new_sid[:8], opening[:30] if opening else "(empty)")
    return _ok({"session_id": new_sid, "message": "会话已创建"})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/message  (SSE 流式响应)
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.post("/message")
def send_message():
    """
    发送用户消息，等待 AI 回复后一次性返回 JSON。

    Body (JSON)
    -----------
    message : str   用户消息文本

    Response
    --------
    { status, reply, turn, cbt_form, timestamp }
    """
    body = request.get_json(silent=True) or {}
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return _err("消息不能为空")

    sid = session.get("session_id")
    if not sid:
        return _err("会话不存在，请先调用 /api/chat/start", 401)

    mgr = get_session_manager()
    try:
        result = mgr.chat(sid, user_msg)
    except KeyError:
        return _err("会话已失效，请刷新页面重新开始", 401)
    except RuntimeError as e:
        return _err(str(e), 500)
    except Exception as e:
        logger.exception("[Route] Unexpected error in /message")
        return _err(f"系统内部错误：{e}", 500)

    return _ok({
        "reply":     result["therapist_reply"],
        "turn":      result["turn"],
        "cbt_form":  result["cbt_form"],
        "timestamp": result["timestamp"],
        "interrupted": result.get("interrupted", False),
        "safety_category": result.get("safety_category"),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/history
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.get("/history")
def get_history():
    """
    返回当前会话的完整对话历史。

    Response
    --------
    { status, history: [{role, content}, ...] }
    """
    sid = session.get("session_id")
    if not sid:
        return _err("会话不存在", 401)
    try:
        history = get_session_manager().get_history(sid)
        return _ok({"history": history})
    except KeyError:
        return _err("会话已失效", 404)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/cbt_form
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.get("/cbt_form")
def get_cbt_form():
    """
    返回当前会话的认知评估表。

    Response
    --------
    { status, cbt_form: {situation, emotion, automatic_thought, cognitive_distortion} }
    """
    sid = session.get("session_id")
    if not sid:
        return _err("会话不存在", 401)
    try:
        form = get_session_manager().get_cbt_form(sid)
        return _ok({"cbt_form": form})
    except KeyError:
        return _err("会话已失效", 404)


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /api/chat/session
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.delete("/session")
def close_session():
    """
    主动结束并销毁当前会话。

    Response
    --------
    { status, message }
    """
    sid = session.pop("session_id", None)
    if sid:
        get_session_manager().close_session(sid)
    return _ok({"message": "会话已结束"})
