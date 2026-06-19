# tests/tracker/test_service_message.py
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from webapp.core.persistence import SessionStore
from webapp.core.tracker_service import TrackerService
from agents.tracker import TrackerNode
from agents.llm_base import LLMClient


class FakeJudge:
    def simple_chat(self, system, user, temperature=None):
        if "核心负面信念" in system:   # conviction prompt
            return '{"conviction": 30, "evidence": "来访者说现在没那么信了"}'
        return '{"narrative": "确信度下降，有好转迹象。", "suggestion": "继续小步行为实验。"}'
    extract_json = staticmethod(LLMClient.extract_json)


class FakeDiag:
    def __call__(self, state):
        return {"cbt_form": {"situation": "看到待办", "emotion": "平静",
                             "automatic_thought": "我什么都做不好", "cognitive_distortion": "以偏概全"}}


def _service(tmp):
    baseline = {"session_id": "base1", "created_at": "2026-06-01T10:00:00",
                "chat_history": [{"role": "user", "content": "看到待办就焦虑，我什么都做不好"}],
                "cbt_form": {"situation": "看到待办", "emotion": "焦虑",
                             "automatic_thought": "我什么都做不好", "cognitive_distortion": "以偏概全"}}
    return TrackerService(
        baseline_loader=lambda sid: baseline,
        tracker_store=SessionStore(tmp, enabled=True),
        question_builder=TrackerNode(llm_client=None),
        judge_llm=FakeJudge(), diagnostician=FakeDiag(), polish=False)


def test_message_advances_then_completes():
    tmp = tempfile.mkdtemp(prefix="trk_")
    svc = _service(tmp)
    start = svc.start_checkin("base1")
    cid, total = start["checkin_id"], start["total"]
    done = None
    for i in range(total):
        done = svc.checkin_message(cid, f"我的回答{i}")
    assert done["done"] is True, done
    assert "report" in done
    rec = SessionStore(tmp, enabled=True).load(cid)
    assert rec["status"] == "completed"
    assert len(rec["chat_history"]) == total * 2


def test_message_returns_next_question_midway():
    tmp = tempfile.mkdtemp(prefix="trk_")
    svc = _service(tmp)
    start = svc.start_checkin("base1")
    res = svc.checkin_message(start["checkin_id"], "第一答")
    assert res["done"] is False and res["question"] and res["q_index"] == 1


if __name__ == "__main__":
    test_message_advances_then_completes()
    test_message_returns_next_question_midway()
    print("OK test_service_message")
