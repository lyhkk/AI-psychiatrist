# tests/tracker/test_node_polish.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.tracker import TrackerNode
from agents.llm_base import LLMClient


class FakeLLM:
    def __init__(self, reply): self._reply = reply; self.calls = 0
    def simple_chat(self, system, user, temperature=None):
        self.calls += 1
        return self._reply(user) if callable(self._reply) else self._reply
    extract_json = staticmethod(LLMClient.extract_json)


def test_good_polish_kept():
    good = "最近再碰到时，「我什么都做不好」这个念头还冒出来吗？现在 0–100 你给它打几分（100=完全信）？"
    node = TrackerNode(llm_client=FakeLLM(good))
    template = ("那时你脑海里会冒出「我什么都做不好」这个想法。最近再遇到类似情况时，这个想法还会出现吗？"
                "如果用 0–100 表示你现在有多相信它（100=完全相信，0=已不相信），你会打几分？")
    out = node.polish_question(template, baseline_form={"automatic_thought": "我什么都做不好"})
    assert out == good


def test_drifting_polish_falls_back():
    bad = "最近还好吗？要不要试试冥想？"
    node = TrackerNode(llm_client=FakeLLM(bad))
    template = ("那时你脑海里会冒出「我什么都做不好」这个想法。最近再遇到类似情况时，这个想法还会出现吗？"
                "如果用 0–100 表示你现在有多相信它（100=完全相信，0=已不相信），你会打几分？")
    out = node.polish_question(template, baseline_form={"automatic_thought": "我什么都做不好"})
    assert out == template  # fell back


def test_build_checkin_questions_no_llm_is_templates():
    node = TrackerNode(llm_client=None)
    form = {"situation": "S", "emotion": "E", "automatic_thought": "T", "cognitive_distortion": "D"}
    assert node.build_checkin_questions(form, polish=True) == node.build_questions(form)


if __name__ == "__main__":
    test_good_polish_kept()
    test_drifting_polish_falls_back()
    test_build_checkin_questions_no_llm_is_templates()
    print("OK test_node_polish")
