# tests/tracker/test_service_start.py
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from webapp.core.persistence import SessionStore
from webapp.core.tracker_service import TrackerService
from agents.tracker import TrackerNode


def _service(tmp):
    baseline = {
        "session_id": "base1", "created_at": "2026-06-01T10:00:00",
        "chat_history": [{"role": "user", "content": "看到待办就焦虑，觉得我什么都做不好"}],
        "cbt_form": {"situation": "看到待办", "emotion": "焦虑",
                     "automatic_thought": "我什么都做不好", "cognitive_distortion": "以偏概全"},
    }
    return TrackerService(
        baseline_loader=lambda sid: baseline if sid == "base1" else (_ for _ in ()).throw(KeyError(sid)),
        tracker_store=SessionStore(tmp, enabled=True),
        question_builder=TrackerNode(llm_client=None),
        judge_llm=None, diagnostician=None,
    )


def test_start_creates_checkin_and_returns_first_question():
    tmp = tempfile.mkdtemp(prefix="trk_")
    svc = _service(tmp)
    res = svc.start_checkin("base1")
    assert res["question"] and res["q_index"] == 0 and res["total"] == 4
    rec = SessionStore(tmp, enabled=True).load(res["checkin_id"])
    assert rec["status"] == "in_progress"
    assert rec["baseline_id"] == "base1"
    assert rec["baseline_form"]["automatic_thought"] == "我什么都做不好"
    assert len(rec["questions"]) == 4
    assert rec["chat_history"][0]["role"] == "assistant"


def test_start_unknown_baseline_raises():
    tmp = tempfile.mkdtemp(prefix="trk_")
    svc = _service(tmp)
    try:
        svc.start_checkin("nope"); raise SystemExit("should raise")
    except KeyError:
        pass


if __name__ == "__main__":
    test_start_creates_checkin_and_returns_first_question()
    test_start_unknown_baseline_raises()
    print("OK test_service_start")
