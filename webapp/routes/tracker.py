# webapp/routes/tracker.py
"""Flask 蓝图：进展追踪器 API（/api/tracker/*）。仅做 HTTP 编解码。"""

from __future__ import annotations

import logging
from flask import Blueprint, Response, jsonify, request

from webapp.core import get_session_manager
from webapp.core.tracker_service import get_tracker_service

logger = logging.getLogger("cbt.webapp.tracker")
tracker_bp = Blueprint("tracker", __name__, url_prefix="/api/tracker")


def _ok(data: dict) -> Response:
    return jsonify({"status": "ok", **data})


def _err(message: str, code: int = 400):
    return jsonify({"status": "error", "message": message}), code


@tracker_bp.get("/baselines")
def baselines():
    records = [get_session_manager().get_record(s["session_id"])
               for s in get_session_manager().list_sessions()]
    return _ok({"baselines": get_tracker_service().list_baselines(records)})


@tracker_bp.post("/start")
def start():
    body = request.get_json(silent=True) or {}
    bid = (body.get("baseline_id") or "").strip()
    if not bid:
        return _err("缺少 baseline_id")
    try:
        return _ok(get_tracker_service().start_checkin(bid))
    except KeyError:
        return _err("基线会话不存在", 404)


@tracker_bp.post("/message")
def message():
    body = request.get_json(silent=True) or {}
    cid = (body.get("checkin_id") or "").strip()
    msg = (body.get("message") or "").strip()
    if not cid or not msg:
        return _err("缺少 checkin_id 或 message")
    try:
        return _ok(get_tracker_service().checkin_message(cid, msg))
    except KeyError:
        return _err("复诊不存在", 404)
    except Exception as e:
        logger.exception("[Tracker] message failed")
        return _err(f"系统内部错误：{e}", 500)


@tracker_bp.get("/checkins")
def checkins():
    bid = request.args.get("baseline_id")
    return _ok({"checkins": get_tracker_service().list_checkins(bid)})


@tracker_bp.get("/report/<cid>")
def report(cid: str):
    try:
        return _ok({"report": get_tracker_service().get_report(cid)})
    except KeyError:
        return _err("复诊不存在", 404)
    except ValueError:
        return _err("复诊尚未完成", 409)
