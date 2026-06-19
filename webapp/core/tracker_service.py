# webapp/core/tracker_service.py
"""
webapp/core/tracker_service.py
进展追踪器服务层：编排复诊与报告计算。

设计：与 Flask 解耦；通过依赖注入接入基线加载器、tracker 存储、问句生成器、
judge 模型、diagnostician，便于单测。复用 eval_pipeline.get_belief_conviction、
DiagnosticianNode、12 类扭曲分类法；不修改任何既有 prompt。
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents import make_initial_state
from .persistence import SessionStore

logger = logging.getLogger("cbt.tracker.service")

_DIMS = [
    ("情境", "situation"), ("情绪", "emotion"),
    ("自动思维", "automatic_thought"), ("认知扭曲", "cognitive_distortion"),
]


def _format_transcript(chat_history: list[dict]) -> str:
    lines = []
    for m in chat_history or []:
        role = "来访者" if m.get("role") == "user" else "咨询师"
        lines.append(f"{role}：{m.get('content', '')}")
    return "\n".join(lines)


# ── 模块级：诊断表 diff ─────────────────────────────────────────────────────
def _diff_forms(before: dict, after: dict) -> list[dict]:
    out = []
    for label, key in _DIMS:
        b = (before or {}).get(key)
        a = (after or {}).get(key)
        if b and a:
            change = "same" if b == a else "changed"
        elif b and not a:
            change = "not_reassessed"
        elif (not b) and a:
            change = "new"
        else:
            change = "same"
        out.append({"dimension": label, "key": key, "before": b, "after": a, "change": change})
    return out


class TrackerService:
    def __init__(self, baseline_loader: Callable[[str], dict],
                 tracker_store: SessionStore, question_builder,
                 judge_llm, diagnostician, polish: bool = True):
        self._baseline_loader = baseline_loader
        self._store = tracker_store
        self._qb = question_builder
        self._judge = judge_llm
        self._diagnostician = diagnostician
        self._polish = polish
        self._lock = threading.Lock()

    # ── 启动复诊 ──────────────────────────────────────────────────────────
    def start_checkin(self, baseline_id: str) -> dict[str, Any]:
        baseline = self._baseline_loader(baseline_id)  # 可抛 KeyError
        baseline_form = dict(baseline.get("cbt_form", {}))
        questions = self._qb.build_checkin_questions(baseline_form, polish=self._polish)
        checkin_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        record = {
            "checkin_id": checkin_id,
            "baseline_id": baseline_id,
            "baseline_created_at": baseline.get("created_at", ""),
            "kind": "tracker_checkin",
            "status": "in_progress",
            "created_at": now,
            "last_active": now,
            "model": os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL") or "",
            "baseline_form": baseline_form,
            "questions": questions,
            "q_index": 0,
            "chat_history": [{"role": "assistant", "content": questions[0]}],
            "report": None,
        }
        self._save(record)
        logger.info("[Tracker] start checkin %s for baseline %s (%d questions)",
                    checkin_id[:8], baseline_id[:8], len(questions))
        return {"checkin_id": checkin_id, "baseline_form": baseline_form,
                "question": questions[0], "q_index": 0, "total": len(questions)}

    # ── 推进问答 ──────────────────────────────────────────────────────────
    def checkin_message(self, checkin_id: str, user_message: str) -> dict[str, Any]:
        with self._lock:
            rec = self._load(checkin_id)
            if rec is None:
                raise KeyError(checkin_id)
            if rec.get("status") == "completed":
                return {"done": True, "report": rec.get("report")}

            history = list(rec.get("chat_history", []))
            history.append({"role": "user", "content": user_message})
            rec["chat_history"] = history
            rec["q_index"] = int(rec.get("q_index", 0)) + 1
            rec["last_active"] = datetime.now().isoformat()

            questions = rec.get("questions", [])
            if rec["q_index"] < len(questions):
                next_q = questions[rec["q_index"]]
                history.append({"role": "assistant", "content": next_q})
                self._save(rec)
                return {"done": False, "question": next_q,
                        "q_index": rec["q_index"], "total": len(questions)}

            report = self.compute_report(rec)
            rec["report"] = report
            rec["status"] = "completed"
            self._save(rec)
            return {"done": True, "report": report}

    # ── 报告计算 ──────────────────────────────────────────────────────────
    def compute_report(self, checkin: dict) -> dict[str, Any]:
        from eval_pipeline import get_belief_conviction  # 复用既有确信度量表

        baseline = self._baseline_loader(checkin["baseline_id"])
        baseline_form = checkin.get("baseline_form", {})

        base_text = _format_transcript(baseline.get("chat_history", []))
        cur_text = _format_transcript(checkin.get("chat_history", []))
        base_conv, base_ev = get_belief_conviction(self._judge, base_text)
        cur_conv, cur_ev = get_belief_conviction(self._judge, cur_text)
        delta = base_conv - cur_conv
        status = "改善" if delta >= 15 else ("恶化" if delta <= -15 else "持平")

        state = make_initial_state()
        state["chat_history"] = list(checkin.get("chat_history", []))
        diag_out = self._diagnostician(state) if self._diagnostician else {}
        current_form = diag_out.get("cbt_form") or make_initial_state()["cbt_form"]

        form_diff = _diff_forms(baseline_form, current_form)
        trend = self._build_trend(checkin, base_conv, cur_conv)
        narrative, suggestion = self._narrate(
            baseline_form, current_form, base_conv, cur_conv, base_ev, cur_ev, status)

        return {
            "baseline_conviction": base_conv, "baseline_evidence": base_ev,
            "current_conviction": cur_conv, "current_evidence": cur_ev,
            "conviction_delta": delta, "status": status,
            "current_form": current_form, "form_diff": form_diff,
            "trend": trend, "narrative": narrative, "suggestion": suggestion,
        }

    def _build_trend(self, checkin: dict, base_conv: int, cur_conv: int) -> list[dict]:
        baseline_id = checkin["baseline_id"]
        others = [r for r in self._store.list_records()
                  if r.get("baseline_id") == baseline_id
                  and r.get("status") == "completed"
                  and r.get("checkin_id") != checkin["checkin_id"]]
        rows = [(r.get("created_at", ""), (r.get("report") or {}).get("current_conviction"))
                for r in others if (r.get("report") or {}).get("current_conviction") is not None]
        rows.append((checkin.get("created_at", ""), cur_conv))
        rows.sort(key=lambda x: x[0])
        pts = [{"label": "基线", "date": checkin.get("baseline_created_at", ""), "conviction": base_conv}]
        for i, (date, conv) in enumerate(rows, start=1):
            pts.append({"label": f"复诊{i}", "date": date, "conviction": conv})
        return pts

    def _narrate(self, baseline_form, current_form, base_conv, cur_conv,
                 base_ev, cur_ev, status) -> tuple[str, str]:
        if self._judge is None:
            return "", ""
        system = (
            "你是一名临床CBT督导，正在对比来访者前后两次的状态变化。\n"
            "信念确信度量表（0–100）：100=深信不疑；50=有所动摇；0=已放弃该负面信念或未体现。\n"
            "整体状态已判定为：{status}（以确信度变化为主）。请仅基于下方已提供的信息，"
            "客观说明各维度变化并引用证据，再给出一条温和的、基于CBT原则的下一步建议。"
            "禁止编造未提供的信息。\n"
            '严格输出 JSON：{{"narrative":"...","suggestion":"..."}}'
        ).format(status=status)
        user = (
            f"基线诊断表：{baseline_form}\n当前诊断表：{current_form}\n"
            f"基线确信度：{base_conv}（依据：{base_ev}）\n"
            f"当前确信度：{cur_conv}（依据：{cur_ev}）"
        )
        try:
            raw = self._judge.simple_chat(system=system, user=user, temperature=0.3)
            parsed = self._judge.extract_json(raw) or {}
            return str(parsed.get("narrative", "")), str(parsed.get("suggestion", ""))
        except Exception as exc:
            logger.warning("[Tracker] narrate failed: %s", exc)
            return "", ""

    # ── 查询 ──────────────────────────────────────────────────────────────
    def list_baselines(self, session_summaries_or_records: list[dict]) -> list[dict]:
        """从会话记录中筛出可作基线者（轮次≥2 或有 automatic_thought）。"""
        out = []
        for r in session_summaries_or_records:
            form = r.get("cbt_form", {}) or {}
            if int(r.get("turn_count", 0) or 0) >= 2 or form.get("automatic_thought"):
                out.append({
                    "session_id": r.get("session_id", ""),
                    "title": (r.get("chat_history") or [{}])[0].get("content", "")[:40]
                             if r.get("chat_history") else r.get("title", "(空会话)"),
                    "last_active": r.get("last_active", ""),
                    "turn_count": int(r.get("turn_count", 0) or 0),
                    "emotion": form.get("emotion"),
                    "cognitive_distortion": form.get("cognitive_distortion"),
                })
        return sorted(out, key=lambda s: s.get("last_active") or "", reverse=True)

    def get_report(self, checkin_id: str) -> dict:
        rec = self._load(checkin_id)
        if rec is None:
            raise KeyError(checkin_id)
        if rec.get("status") != "completed" or not rec.get("report"):
            raise ValueError("复诊尚未完成")
        return rec["report"]

    def list_checkins(self, baseline_id: str | None = None) -> list[dict]:
        rows = []
        for r in self._store.list_records():
            if baseline_id and r.get("baseline_id") != baseline_id:
                continue
            rows.append({
                "checkin_id": r.get("checkin_id", r.get("session_id", "")),
                "baseline_id": r.get("baseline_id", ""),
                "status": r.get("status", ""),
                "created_at": r.get("created_at", ""),
                "current_conviction": (r.get("report") or {}).get("current_conviction"),
                "status_label": (r.get("report") or {}).get("status"),
            })
        return sorted(rows, key=lambda s: s.get("created_at") or "", reverse=True)

    # ── 内部 ──────────────────────────────────────────────────────────────
    def _save(self, record: dict) -> None:
        # SessionStore 以 record["session_id"] 命名文件；令其等于 checkin_id 复用存储
        record["session_id"] = record["checkin_id"]
        self._store.save(record)

    def _load(self, checkin_id: str) -> dict | None:
        return self._store.load(checkin_id)


# ── 应用级单例 ──────────────────────────────────────────────────────────────
_tracker_instance: "TrackerService | None" = None


def get_tracker_service() -> "TrackerService":
    global _tracker_instance
    if _tracker_instance is None:
        from agents.llm_base import LLMClient
        from agents import DiagnosticianNode
        from agents.tracker import TrackerNode
        from .session_manager import get_session_manager
        directory = os.getenv("TRACKER_STORE_DIR") or str(_PROJECT_ROOT / "results" / "trackers")
        polish = os.getenv("ENABLE_TRACKER_POLISH", "true").strip().lower() in ("1", "true", "yes", "on")
        _tracker_instance = TrackerService(
            baseline_loader=get_session_manager().get_record,
            tracker_store=SessionStore(directory, enabled=True),
            question_builder=TrackerNode(LLMClient.from_role("therapist") if polish else None),
            judge_llm=LLMClient.from_role("judge"),
            diagnostician=DiagnosticianNode(),
            polish=polish,
        )
        logger.info("[Tracker] service initialized  dir=%s  polish=%s", directory, polish)
    return _tracker_instance
