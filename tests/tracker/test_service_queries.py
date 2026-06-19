# tests/tracker/test_service_queries.py
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from webapp.core.persistence import SessionStore
from webapp.core.tracker_service import TrackerService
from agents.tracker import TrackerNode


def test_list_baselines_filters_thin_sessions():
    tmp = tempfile.mkdtemp(prefix="trk_")
    sessions = {
        "rich": {"session_id": "rich", "turn_count": 3, "last_active": "2026-06-02",
                 "chat_history": [{"role": "user", "content": "我什么都做不好"}],
                 "cbt_form": {"automatic_thought": "我什么都做不好", "emotion": "焦虑",
                              "cognitive_distortion": "以偏概全", "situation": "S"}},
        "thin": {"session_id": "thin", "turn_count": 0, "last_active": "2026-06-03",
                 "chat_history": [], "cbt_form": {}},
    }
    svc = TrackerService(baseline_loader=lambda s: sessions[s],
                         tracker_store=SessionStore(tmp, enabled=True),
                         question_builder=TrackerNode(None), judge_llm=None, diagnostician=None)
    bl = svc.list_baselines(list(sessions.values()))
    ids = [b["session_id"] for b in bl]
    assert "rich" in ids and "thin" not in ids


if __name__ == "__main__":
    test_list_baselines_filters_thin_sessions()
    print("OK test_service_queries")
