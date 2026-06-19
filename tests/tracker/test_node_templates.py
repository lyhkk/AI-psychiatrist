# tests/tracker/test_node_templates.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.tracker import TrackerNode


def test_full_form_yields_four_questions():
    node = TrackerNode(llm_client=None)
    form = {
        "situation": "看到待办事项",
        "emotion": "焦虑",
        "automatic_thought": "我什么都做不好",
        "cognitive_distortion": "以偏概全",
    }
    qs = node.build_questions(form)
    assert len(qs) == 4, qs
    assert "看到待办事项" in qs[0]
    assert "我什么都做不好" in qs[1]
    assert "0" in qs[1] and "100" in qs[1]
    assert "焦虑" in qs[2]
    assert qs[3]  # coping question always present


def test_empty_form_yields_only_coping_question():
    node = TrackerNode(llm_client=None)
    qs = node.build_questions({"situation": None, "emotion": None,
                               "automatic_thought": None, "cognitive_distortion": None})
    assert len(qs) == 1, qs
    assert "应对" in qs[0]  # the coping fallback question


if __name__ == "__main__":
    test_full_form_yields_four_questions()
    test_empty_form_yields_only_coping_question()
    print("OK test_node_templates")
