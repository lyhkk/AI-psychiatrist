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

    # ── 内部 ──────────────────────────────────────────────────────────────
    def _save(self, record: dict) -> None:
        # SessionStore 以 record["session_id"] 命名文件；令其等于 checkin_id 复用存储
        record["session_id"] = record["checkin_id"]
        self._store.save(record)

    def _load(self, checkin_id: str) -> dict | None:
        return self._store.load(checkin_id)
