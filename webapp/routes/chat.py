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
POST   /api/chat/start                     开始新的 CBT 会话
POST   /api/chat/message                   发送消息，获取治疗师回复
GET    /api/chat/history                   获取当前会话的完整对话历史
GET    /api/chat/cbt_form                  获取当前认知评估表
DELETE /api/chat/session                   结束并销毁当前会话
GET    /api/chat/sessions                  会话历史列表（全部会话摘要）
GET    /api/chat/sessions/<sid>            单条完整会话记录
GET    /api/chat/sessions/<sid>/export     导出会话（format=json|md）
POST   /api/chat/resume                    切换到并继续某段历史会话
POST   /api/chat/undo                      撤销当前会话最近一轮
"""

from __future__ import annotations

import json
import logging
from flask import (
    Blueprint,
    Response,
    jsonify,
    request,
    session,
)

from webapp.core import get_session_manager
from webapp.core.export import to_markdown

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


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/sessions  —— 会话历史列表
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.get("/sessions")
def list_sessions():
    """
    返回服务器上全部会话的摘要列表（按最后活跃倒序）。

    Response
    --------
    { status, sessions: [{session_id, created_at, last_active, turn_count,
                          title, emotion, cognitive_distortion}, ...] }
    """
    sessions = get_session_manager().list_sessions()
    return _ok({"sessions": sessions})


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/sessions/<sid>  —— 单条完整记录
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.get("/sessions/<sid>")
def get_session_record(sid: str):
    """返回指定会话的完整记录（含对话历史与诊断演变）。"""
    try:
        record = get_session_manager().get_record(sid)
        return _ok({"session": record})
    except KeyError:
        return _err("会话不存在", 404)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/sessions/<sid>/export?format=json|md  —— 下载
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.get("/sessions/<sid>/export")
def export_session(sid: str):
    """
    导出指定会话为可下载文件。

    Query
    -----
    format : "json"（默认）| "md"
    """
    fmt = (request.args.get("format") or "json").lower()
    try:
        record = get_session_manager().get_record(sid)
    except KeyError:
        return _err("会话不存在", 404)

    short = sid[:8]
    if fmt == "json":
        body = json.dumps(record, ensure_ascii=False, indent=2)
        return Response(
            body,
            mimetype="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="cbt_session_{short}.json"'
            },
        )
    if fmt in ("md", "markdown"):
        body = to_markdown(record)
        return Response(
            body,
            mimetype="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="cbt_session_{short}.md"'
            },
        )
    return _err("不支持的导出格式，仅支持 json 或 md")


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/resume  —— 继续某段历史会话
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.post("/resume")
def resume_session():
    """
    将当前浏览器会话切换到指定历史会话，之后 /message 在其上追加续聊。

    Body (JSON)
    -----------
    session_id : str

    Response
    --------
    { status, session_id, turn, history, cbt_form }
    """
    body = request.get_json(silent=True) or {}
    sid = (body.get("session_id") or "").strip()
    if not sid:
        return _err("缺少 session_id")

    mgr = get_session_manager()
    try:
        record = mgr.get_record(sid)
    except KeyError:
        return _err("会话不存在", 404)

    session["session_id"] = sid
    logger.info("[Route] /resume  -> session=%s  turn=%s", sid[:8], record.get("turn_count"))
    return _ok({
        "session_id": sid,
        "turn":       record.get("turn_count", 0),
        "history":    record.get("chat_history", []),
        "cbt_form":   record.get("cbt_form", {}),
    })


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/undo  —— 撤销当前会话最近一轮
# ─────────────────────────────────────────────────────────────────────────────

@chat_bp.post("/undo")
def undo_turn():
    """
    撤销当前会话最近一轮对话。

    Response
    --------
    { status, turn, history, cbt_form }
    """
    sid = session.get("session_id")
    if not sid:
        return _err("会话不存在，请先开始或选择一个会话", 401)
    mgr = get_session_manager()
    try:
        result = mgr.undo_last_turn(sid)
    except KeyError:
        return _err("会话已失效", 404)
    except ValueError as e:
        return _err(str(e))
    return _ok({
        "turn":     result["turn"],
        "history":  result["history"],
        "cbt_form": result["cbt_form"],
    })
