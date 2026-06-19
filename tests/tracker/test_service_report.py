# tests/tracker/test_service_report.py
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from webapp.core.persistence import SessionStore
from webapp.core.tracker_service import TrackerService, _diff_forms
from agents.tracker import TrackerNode
from agents.llm_base import LLMClient


class FakeJudge:
    """基线确信度 80，复诊确信度 30 -> delta=50 -> 改善。"""
    def __init__(self): self.n = 0
    def simple_chat(self, system, user, temperature=None):
        if "核心负面信念" in system:   # conviction prompt
            self.n += 1
            return '{"conviction": 80, "evidence": "基线很信"}' if self.n == 1 \
                else '{"conviction": 30, "evidence": "现在没那么信"}'
        return '{"narrative": "好转。", "suggestion": "继续。"}'
    extract_json = staticmethod(LLMClient.extract_json)


class FakeDiag:
    def __call__(self, state):
        return {"cbt_form": {"situation": "看到待办", "emotion": "平静",
                             "automatic_thought": "我什么都做不好", "cognitive_distortion": None}}


def test_diff_forms_change_types():
    before = {"situation": "S", "emotion": "焦虑", "automatic_thought": "T", "cognitive_distortion": "以偏概全"}
    after = {"situation": "S", "emotion": "平静", "automatic_thought": "T", "cognitive_distortion": None}
    diff = {d["key"]: d["change"] for d in _diff_forms(before, after)}
    assert diff["situation"] == "same"
    assert diff["emotion"] == "changed"
    assert diff["automatic_thought"] == "same"
    assert diff["cognitive_distortion"] == "not_reassessed"


def test_report_conviction_delta_and_status():
    tmp = tempfile.mkdtemp(prefix="trk_")
    baseline = {"session_id": "base1", "created_at": "2026-06-01T10:00:00",
                "chat_history": [{"role": "user", "content": "我什么都做不好"}],
                "cbt_form": {"situation": "看到待办", "emotion": "焦虑",
                             "automatic_thought": "我什么都做不好", "cognitive_distortion": "以偏概全"}}
    svc = TrackerService(baseline_loader=lambda sid: baseline,
                         tracker_store=SessionStore(tmp, enabled=True),
                         question_builder=TrackerNode(llm_client=None),
                         judge_llm=FakeJudge(), diagnostician=FakeDiag(), polish=False)
    start = svc.start_checkin("base1")
    res = None
    for i in range(start["total"]):
        res = svc.checkin_message(start["checkin_id"], f"答{i}")
    rep = res["report"]
    assert rep["baseline_conviction"] == 80
    assert rep["current_conviction"] == 30
    assert rep["conviction_delta"] == 50
    assert rep["status"] == "改善"
    assert rep["narrative"] and rep["suggestion"]
    assert len(rep["trend"]) == 2
    assert rep["trend"][0]["conviction"] == 80 and rep["trend"][1]["conviction"] == 30


if __name__ == "__main__":
    test_diff_forms_change_types()
    test_report_conviction_delta_and_status()
    print("OK test_service_report")
