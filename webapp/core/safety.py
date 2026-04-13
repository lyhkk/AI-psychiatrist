"""
webapp/core/safety.py

对话安全检测服务。

职责：
- 检测用户输入中的严重心理危机信号，必要时中断当前轮对话并返回关怀告警信息。
- 检测大模型输出中的危险、攻击、教唆、自伤引导等不安全内容，并替换为安全回复。

设计原则：
- 规则优先，保持可解释、低依赖。
- 仅实现当前系统所需的最小能力，不引入额外模型或复杂审核链。
"""

from __future__ import annotations

from dataclasses import dataclass


CARE_HOTLINE_MESSAGE = (
    "我注意到你提到了可能涉及严重自伤、自杀或其他紧急风险的内容。"
    "我现在不能继续常规对话。请立即联系现实中的紧急支持：\n"
    "- 中国生命危机干预热线：400-161-9995\n"
    "- 北京心理危机干预热线：800-810-1117（座机） / 010-8295-1332（手机）\n"
    "- 如存在现实中的紧急危险，请立即拨打 110 或尽快前往最近医院急诊。\n"
    "如果你愿意，也请立刻联系你身边值得信任的家人、朋友、老师或同事，让他们现在陪着你。"
)

SAFE_FALLBACK_MESSAGE = (
    "抱歉，我不能提供包含攻击、辱骂、危险教唆或可能伤害他人的内容。"
    "如果你愿意，我们可以继续用安全、尊重的方式讨论你的感受，"
    "或一起梳理当前让你最难受的事情。"
)

_CRISIS_KEYWORDS = [
    "自杀", "想死", "不想活", "活不下去", "结束生命", "了结自己", "轻生",
    "割腕", "上吊", "跳楼", "服毒", "吞药", "烧炭", "遗书", "自残", "伤害自己",
    "杀了自己", "去死", "我会死", "我今晚就", "不如死了", "没有活着的意义",
]

_OUTPUT_BLOCK_KEYWORDS = [
    "去死", "该死", "废物", "蠢货", "垃圾", "贱", "弄死", "杀了他", "报复他",
    "教训他", "怎么自杀", "自杀方法", "结束生命的方法", "最痛快的死法", "伤害别人",
    "捅", "投毒", "炸", "勒死", "毁掉他", "怂恿", "你应该去",
]


@dataclass(frozen=True)
class SafetyCheckResult:
    flagged: bool
    category: str | None = None
    message: str | None = None


class SafetyService:
    """规则型安全检测服务。"""

    def check_user_message(self, text: str) -> SafetyCheckResult:
        normalized = self._normalize(text)
        if self._contains_any(normalized, _CRISIS_KEYWORDS):
            return SafetyCheckResult(
                flagged=True,
                category="crisis",
                message=CARE_HOTLINE_MESSAGE,
            )
        return SafetyCheckResult(flagged=False)

    def check_model_output(self, text: str) -> SafetyCheckResult:
        normalized = self._normalize(text)
        if self._contains_any(normalized, _CRISIS_KEYWORDS):
            return SafetyCheckResult(
                flagged=True,
                category="crisis_output",
                message=CARE_HOTLINE_MESSAGE,
            )
        if self._contains_any(normalized, _OUTPUT_BLOCK_KEYWORDS):
            return SafetyCheckResult(
                flagged=True,
                category="unsafe_output",
                message=SAFE_FALLBACK_MESSAGE,
            )
        return SafetyCheckResult(flagged=False)

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(str(text or "").lower().split())

    @staticmethod
    def _contains_any(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

