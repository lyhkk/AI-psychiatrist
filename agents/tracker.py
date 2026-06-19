# agents/tracker.py
"""
agents/tracker.py
进展追踪器问句生成节点（TrackerNode）。

职责：据基线 CBT 评估表生成"复诊"问句。
  1. build_questions：确定性模板（唯一信源，零自创），仅就非空字段生成。
  2. polish_question：可选的受限 LLM 润色（仅改措辞），带校验+回退原模板。
不修改任何既有 prompt；信念再评分沿用 0–100 量表语义，扭曲分类沿用 diagnostician 分类法。
"""

from __future__ import annotations

import logging

from .llm_base import LLMClient

logger = logging.getLogger("cbt.tracker")

_POLISH_SYSTEM = """\
你是一名 CBT 随访助手。下面是一条已基于来访者基线认知评估表生成的"复诊问句模板"。
你的唯一任务：在【不改变其含义、不新增任何话题或建议、不删除其引用的基线内容】的前提下，
把它润色得更自然、温和、口语化。
- 必须原样保留问句中引用的基线片段（情境/自动思维原文）与 0–100 评分要求（若模板含）。
- 不得加入模板之外的新问题、解释或建议。
只输出润色后的一句话，不要引号、不要任何解释。
"""


class TrackerNode:
    """复诊问句生成器。llm_client 为 None 时仅用模板（不润色）。"""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client

    def build_questions(self, baseline_form: dict) -> list[str]:
        """确定性模板，仅就非空字段生成；末尾恒含应对回顾问句。"""
        situation = (baseline_form or {}).get("situation")
        emotion = (baseline_form or {}).get("emotion")
        thought = (baseline_form or {}).get("automatic_thought")

        # 注：cognitive_distortion 不单独成问句——其变化在报告阶段对比，
        # 复诊只重探可观测的 情境/自动思维/情绪/应对。
        questions: list[str] = []
        if situation:
            questions.append(
                f"上次你提到在「{situation}」时感到困扰。最近有没有再遇到类似的情境？当时发生了什么？"
            )
        if thought:
            questions.append(
                f"那时你脑海里会冒出「{thought}」这个想法。最近再遇到类似情况时，这个想法还会出现吗？"
                f"如果用 0–100 表示你现在有多相信它（100=完全相信，0=已不相信），你会打几分？"
            )
        if emotion:
            questions.append(
                f"再遇到这种情况时，你的情绪和上次的「{emotion}」相比，有什么变化吗？"
            )
        # 应对回顾：恒定保底，确保复诊至少一问
        questions.append("这段时间，你有没有尝试一些新的方式去应对它？效果怎么样？")
        return questions

    # ── 受限润色 ────────────────────────────────────────────────────────────
    def polish_question(self, template: str, baseline_form: dict) -> str:
        """对单条模板做受限润色；任一校验失败则回退原模板。"""
        if self.llm is None:
            return template
        try:
            polished = self.llm.simple_chat(
                system=_POLISH_SYSTEM, user=template, temperature=0.3
            ).strip()
        except Exception as exc:
            logger.warning("[Tracker] polish LLM failed: %s", exc)
            return template
        if not self._polish_valid(template, polished, baseline_form):
            logger.info("[Tracker] polish rejected, using template")
            return template
        return polished

    @staticmethod
    def _polish_valid(template: str, polished: str, baseline_form: dict) -> bool:
        if not polished or len(polished) > len(template) * 1.8:
            return False
        # 必须保留模板引用的基线原文（情境/自动思维）
        for key in ("situation", "automatic_thought"):
            val = (baseline_form or {}).get(key)
            if val and val in template and val not in polished:
                return False
        # 模板含评分要求则润色须保留 0 与 100
        if "0" in template and "100" in template:
            if "0" not in polished or "100" not in polished:
                return False
        return True

    def build_checkin_questions(self, baseline_form: dict, polish: bool = True) -> list[str]:
        """生成复诊问句：模板兜底 +（可选）逐条受限润色。"""
        templates = self.build_questions(baseline_form)
        if not polish or self.llm is None:
            return templates
        return [self.polish_question(t, baseline_form) for t in templates]
