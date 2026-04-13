"""
eval_pipeline.py
Step 4 — 评估管线。

模块一：基于「事实证据清晰度熵减」的信息增益（IG-PQA）计算。
        测算标的：患者对自身负面情绪提供的客观事实证据的清晰程度（而非认知扭曲分类）。
模块二：调用 LLM 进行 CTRS 0-6 分评估，严格 JSON 解析输出。
模块三：信念确信度（Belief Conviction）衰减指标逐轮评测。
        测算标的：患者对核心负面信念的确信程度（0-100），Score = Conviction前 - Conviction后。
模块四：全流程评估入口，读取 run_simulation.py 生成的 transcript JSON，
        输出完整评估报告。

用法示例：
    # 评测单个文件
    python eval_pipeline.py --transcript results/sim/cbt-discover/psyqa0_xxx.json

    # 评测多个文件（空格分隔）
    python eval_pipeline.py --transcript results/sim/cbt-discover/psyqa0_xxx.json results/sim/cbt-discover/psyqa1_xxx.json

    # 评测整个目录下所有 JSON
    python eval_pipeline.py --transcript results/sim/cbt-discover/

    # 指定输出目录
    python eval_pipeline.py --transcript results/sim/cbt-discover/ --output-dir results/eval/cbt-discover/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

try:
    from scipy.stats import entropy as scipy_entropy
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

sys.path.insert(0, str(Path(__file__).parent))
from agents.llm_base import LLMClient

# ─────────────────────────────────────────────────────────────────────────────
# 日志：同时输出到控制台和 logs/eval_<timestamp>.log
# ─────────────────────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

def _setup_logger() -> logging.Logger:
    from datetime import datetime
    log_file = _LOG_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console_handler, file_handler],
    )
    root_logger = logging.getLogger("cbt.eval_pipeline")
    root_logger.info("[Log] Session log file: %s", log_file)
    return root_logger


logger = _setup_logger()



# ─────────────────────────────────────────────────────────────────────────────
# 模块一：信息增益（IG-PQA）计算
# ─────────────────────────────────────────────────────────────────────────────

_ENTROPY_SYSTEM = """\
你是一个临床CBT督导。请根据提供的对话历史，评估来访者对自身负面情绪所提供的
客观事实证据的清晰程度（例如：发生了什么具体事件、时间、地点、客观后果）。

请将"事实证据清晰度"划分为以下5个维度，并以JSON格式输出每个维度当前的概率权重
（权重代表来访者的表述集中在该层级的可能性，总和为1.0）：
  "完全模糊"     : 仅有宏观抱怨或情绪宣泄，无任何具体事实（例如：'我就是很差劲'）
  "轻度具体"     : 提及了大致情境，但缺少时间/地点/客观后果（例如：'面试失败了'）
  "中度具体"     : 有明确事件和部分细节，但因果链不完整
  "高度具体"     : 提供了完整的事件描述，包括时间、地点、经过和客观后果
  "反事实澄清"   : 来访者主动区分了主观解读与客观事实，显现认知松动迹象

输出格式示例：
{"完全模糊": 0.6, "轻度具体": 0.3, "中度具体": 0.1, "高度具体": 0.0, "反事实澄清": 0.0}
只输出JSON对象，不要任何额外文字。
"""


def _shannon_entropy(prob_dict: dict[str, float]) -> float:
    """计算概率分布的香农熵（bits）。"""
    probs = [float(v) for v in prob_dict.values()]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs if p > 0]
    if _SCIPY_OK:
        return float(scipy_entropy(probs, base=2))
    return float(-sum(p * math.log2(p) for p in probs))


# 事实证据清晰度5个维度的均匀分布（回退用）
_CLARITY_LABELS = ["完全模糊", "轻度具体", "中度具体", "高度具体", "反事实澄清"]
_UNIFORM_CLARITY = {lbl: 1.0 / len(_CLARITY_LABELS) for lbl in _CLARITY_LABELS}


def get_clarity_distribution(
    llm: LLMClient,
    conversation_text: str,
) -> dict[str, float]:
    """
    调用 LLM，返回当前对话下「事实证据清晰度」5个维度的概率分布。
    若调用失败，退回均匀分布（最大熵，代表最不确定）。
    """
    try:
        raw = llm.simple_chat(
            system=_ENTROPY_SYSTEM,
            user=f"以下是对话记录：\n\n{conversation_text}\n\n请输出事实证据清晰度概率JSON。",
            temperature=0.0,
        )
        parsed = llm.extract_json(raw)
        if parsed and isinstance(parsed, dict):
            clean = {k: float(v) for k, v in parsed.items()
                     if isinstance(v, (int, float)) and float(v) >= 0}
            if clean:
                return clean
    except Exception as exc:
        logger.warning("[IG] clarity distribution call failed: %s", exc)
    return dict(_UNIFORM_CLARITY)


def compute_information_gain(
    llm: LLMClient,
    conv_before: str,
    conv_after: str,
) -> float:
    """
    计算治疗师一次苏格拉底提问带来的信息增益（策略A）。

    测算标的：患者对自身负面情绪提供的「客观事实证据清晰度」的熵减。
    IG = H(clarity_before) - H(clarity_after)
    H : Shannon 熵（bits）
    IG > 0 表示提问有效将患者从「模糊宏观抱怨」拉向「清晰微观事实」。

    Parameters
    ----------
    conv_before : 治疗师提问前的对话文本
    conv_after  : 患者回答后的对话文本

    Returns
    -------
    float : 信息增益值（bits）
    """
    p_before = get_clarity_distribution(llm, conv_before)
    p_after = get_clarity_distribution(llm, conv_after)

    h_before = _shannon_entropy(p_before)
    h_after = _shannon_entropy(p_after)
    ig = h_before - h_after

    logger.debug(
        "[IG] H_before=%.4f bits  H_after=%.4f bits  IG=%.4f bits",
        h_before, h_after, ig,
    )
    return ig


def eval_ig_pqa_from_transcript(
    llm: LLMClient,
    transcript: list[dict],
) -> dict:
    """
    从完整对话记录中逐轮计算 IG-PQA。

    逻辑：每当出现 therapist -> patient 相邻轮次，
    计算 IG(therapist 提问前 vs patient 回答后)。

    Returns
    -------
    {
        "ig_list"              : list[float],
        "ig_mean"              : float,
        "ig_positive_ratio"    : float,  # IG > 0 的比例
        "high_value_turns"     : int,    # IG > 0 的轮数
        "total_therapist_turns": int,
    }
    """
    ig_list: list[float] = []
    conv_so_far: list[str] = []

    i = 0
    while i < len(transcript):
        entry = transcript[i]
        role = entry.get("role", "")
        content = entry.get("content", "")

        if role == "patient":
            conv_so_far.append(f"来访者：{content}")
        elif role == "therapist":
            # conv_before = 治疗师发言之前
            conv_before = "\n".join(conv_so_far)
            conv_so_far.append(f"咨询师：{content}")

            # 查找紧接着的患者回复
            if i + 1 < len(transcript) and transcript[i + 1].get("role") == "patient":
                next_patient = transcript[i + 1].get("content", "")
                conv_after = conv_before + f"\n咨询师：{content}\n来访者：{next_patient}"
                ig = compute_information_gain(llm, conv_before, conv_after)
                ig_list.append(ig)
                logger.info(
                    "[IG-PQA] Turn %s | IG=%.4f bits | %s",
                    entry.get("turn", "?"), ig,
                    "✓ 高价值" if ig > 0 else "✗ 低价值",
                )
        i += 1

    n = len(ig_list)
    ig_mean = sum(ig_list) / n if n else 0.0
    high_value = sum(1 for v in ig_list if v > 0)
    ig_pos_ratio = high_value / n if n else 0.0

    logger.info(
        "[IG-PQA] mean=%.4f bits | positive_ratio=%.1f%% | high_value=%d/%d",
        ig_mean, ig_pos_ratio * 100, high_value, n,
    )
    return {
        "ig_list": ig_list,
        "ig_mean": ig_mean,
        "ig_positive_ratio": ig_pos_ratio,
        "high_value_turns": high_value,
        "total_therapist_turns": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 模块二：CTRS 0-6 分评估（LLM-as-a-Judge）
# ─────────────────────────────────────────────────────────────────────────────

_CTRS_SYSTEM = """\
你是一名资深CBT临床督导。请阅读以下完整的多轮心理咨询对话记录。
请根据认知疗法评定量表（CTRS）的以下三个核心维度，对咨询师的整体表现进行0-6分的严格打分：
  0=破坏性  1=劣质  2=较差  3=合格  4=良好  5=优秀  6=杰出

1. understanding（理解与共情）：
   咨询师是否准确捕捉到了来访者的内部现实，且没有虚假的无底线迎合？

2. guided_discovery（引导式发现）：
   咨询师是否有效使用了苏格拉底式提问，引导来访者产生认知顿悟，而非直接说教？

3. interpersonal_effectiveness（人际效能）：
   咨询师是否展现了真实的温暖和专业性，并成功化解了来访者的防御？

请输出严格的JSON格式，justification 中引用对话原句作为依据：
{
  "understanding": <0-6整数>,
  "guided_discovery": <0-6整数>,
  "interpersonal_effectiveness": <0-6整数>,
  "justification": "<引用原文的具体说明>"
}
只输出JSON，不要任何额外文字。
"""


def _format_transcript(transcript: list[dict]) -> str:
    """将 transcript 列表格式化为可读对话文本。"""
    lines = []
    for entry in transcript:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if role == "therapist":
            lines.append(f"咨询师：{content}")
        elif role == "patient":
            lines.append(f"来访者：{content}")
    return "\n".join(lines)


def eval_ctrs(
    llm: LLMClient,
    transcript: list[dict],
) -> dict:
    """
    对完整对话记录进行 CTRS 三维度 0-6 分评估。

    Parameters
    ----------
    llm        : LLMClient 实例（建议使用能力强的模型，如 gpt-4o / deepseek）
    transcript : run_simulation.py 生成的 transcript 列表

    Returns
    -------
    {
        "understanding"              : int (0-6),
        "guided_discovery"           : int (0-6),
        "interpersonal_effectiveness": int (0-6),
        "ctrs_avg"                   : float,
        "justification"              : str,
        "raw_response"               : str,
    }
    """
    transcript_text = _format_transcript(transcript)
    turn_count = sum(1 for e in transcript if e.get("role") == "therapist")
    user_prompt = (
        f"以下是完整的心理咨询对话记录（共 {turn_count} 轮治疗师发言）：\n\n"
        f"{transcript_text}\n\n请按CTRS量表要求评分并输出JSON。"
    )

    try:
        raw = llm.simple_chat(
            system=_CTRS_SYSTEM,
            user=user_prompt,
            temperature=0.0,
        )
        # 使用宽松正则匹配最外层完整 JSON（justification 内可能含花括号）
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = None
    except Exception as exc:
        logger.error("[CTRS] LLM call or JSON parse failed: %s", exc)
        parsed = None
        raw = ""

    if parsed:
        keys = ["understanding", "guided_discovery", "interpersonal_effectiveness"]
        scores = {k: int(parsed.get(k, 0)) for k in keys}
        scores["ctrs_avg"] = sum(scores[k] for k in keys) / len(keys)
        scores["justification"] = parsed.get("justification", "")
        scores["raw_response"] = raw
        logger.info(
            "[CTRS] understanding=%d  guided_discovery=%d  interpersonal=%d  avg=%.2f",
            scores["understanding"], scores["guided_discovery"],
            scores["interpersonal_effectiveness"], scores["ctrs_avg"],
        )
        return scores
    else:
        logger.warning("[CTRS] Failed to parse CTRS scores, returning zeros")
        return {
            "understanding": 0,
            "guided_discovery": 0,
            "interpersonal_effectiveness": 0,
            "ctrs_avg": 0.0,
            "justification": "解析失败",
            "raw_response": raw if 'raw' in dir() else "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 模块三：信念确信度（Belief Conviction）衰减指标（策略B）
# ─────────────────────────────────────────────────────────────────────────────

_CONVICTION_SYSTEM = """\
你是一名临床CBT督导。请根据以下对话记录，评估来访者目前对其核心负面信念的确信程度。

核心负面信念是指来访者反复暗示的自我否定性判断，例如：
  "我是一个废物/失败者/没用的人"、"我永远不会成功"、"没有人关心我"等。

请输出一个 0 到 100 之间的整数：
  100 = 来访者对负面信念深信不疑，完全封闭，不接受任何质疑
   75 = 来访者坚定持有负面信念，偶有犹豫但立刻收回
   50 = 来访者对负面信念有所动摇，能听进部分质疑
   25 = 来访者开始怀疑负面信念，开放度明显提升
    0 = 来访者已完全放弃核心负面信念，或对话中未体现明显负面信念

同时简要说明判断依据（引用对话原句）。

输出严格JSON格式：
{"conviction": <0-100整数>, "evidence": "<引用原文的判断依据>"}
只输出JSON对象，不要任何额外文字。
"""


def get_belief_conviction(
    llm: LLMClient,
    conversation_text: str,
) -> tuple[int, str]:
    """
    调用 LLM 评估当前对话中患者对核心负面信念的确信程度（0-100）。

    Returns
    -------
    (conviction_score, evidence_text)
    失败时返回 (50, "解析失败")，使用中位数避免偏向任一方向。
    """
    try:
        raw = llm.simple_chat(
            system=_CONVICTION_SYSTEM,
            user=f"以下是对话记录：\n\n{conversation_text}\n\n请评估来访者的信念确信度并输出JSON。",
            temperature=0.0,
        )
        parsed = llm.extract_json(raw)
        if parsed and isinstance(parsed, dict):
            score = int(parsed.get("conviction", 50))
            score = max(0, min(100, score))  # 夹紧到 [0, 100]
            evidence = str(parsed.get("evidence", ""))
            return score, evidence
    except Exception as exc:
        logger.warning("[Conviction] LLM call failed: %s", exc)
    return 50, "解析失败"


def eval_conviction_from_transcript(
    llm: LLMClient,
    transcript: list[dict],
) -> dict:
    """
    从完整对话记录中逐轮计算信念确信度（Belief Conviction）衰减指标。

    采用「峰值衰减法」（Peak-to-Current Decay）：
      Score(t) = Max_Conviction_So_Far - Conviction_after(t)

    设计逻辑：
      - 信念在被暴露初期确信度会上升（正常现象），不应被惩罚。
      - 只奖励从历史最高点开始的实质性动摇。
      - 历史最高确信度（peak）随每轮 conviction_after 动态更新。

    示例：
      T1 患者爆发：conviction_after=90，peak=90，Score = 90-90 = 0
      T2 面质后：  conviction_after=50，peak=90，Score = 90-50 = +40  ✓ 高价值
      T3 防御反弹：conviction_after=70，peak=90，Score = 90-70 = +20  ✓ 仍有净衰减

    Returns
    -------
    {
        "conviction_list"       : list[dict],  # 每轮详情
        "score_list"            : list[int],   # 每轮峰值衰减得分
        "score_mean"            : float,       # 平均峰值衰减得分
        "positive_ratio"        : float,       # Score > 0 的比例
        "effective_turns"       : int,         # Score > 0 的轮数
        "total_therapist_turns" : int,
        "conviction_peak"       : int,         # 全程最高确信度
        "conviction_end"        : int,         # 最终确信度
        "peak_to_end_decay"     : int,         # 峰值到末尾的总衰减（正为好转）
    }
    """
    conviction_list: list[dict] = []
    score_list: list[int] = []
    conv_so_far: list[str] = []
    peak: int = 0  # 历史最高确信度，随对话动态更新

    i = 0
    while i < len(transcript):
        entry = transcript[i]
        role = entry.get("role", "")
        content = entry.get("content", "")

        if role == "patient":
            conv_so_far.append(f"来访者：{content}")
        elif role == "therapist":
            conv_before = "\n".join(conv_so_far)
            conv_so_far.append(f"咨询师：{content}")

            if i + 1 < len(transcript) and transcript[i + 1].get("role") == "patient":
                next_patient = transcript[i + 1].get("content", "")
                conv_after = "\n".join(conv_so_far) + f"\n来访者：{next_patient}"

                # 治疗师提问前的确信度（用于记录，不用于计分）
                score_before, evidence_before = get_belief_conviction(llm, conv_before)
                # 患者回答后的确信度
                score_after, evidence_after = get_belief_conviction(llm, conv_after)

                # 更新历史峰值（两者取高）
                peak = max(peak, score_before, score_after)

                # 峰值衰减得分：峰值 - 当前，正数 = 从峰值下降，负数 = 仍在攀升
                peak_decay_score = peak - score_after

                score_list.append(peak_decay_score)
                turn_no = entry.get("turn", "?")
                conviction_list.append({
                    "turn": turn_no,
                    "conviction_before": score_before,
                    "conviction_after": score_after,
                    "peak_so_far": peak,
                    "peak_decay_score": peak_decay_score,
                    "evidence_before": evidence_before,
                    "evidence_after": evidence_after,
                })
                logger.info(
                    "[Conviction] Turn %s | before=%d -> after=%d | peak=%d | score=%+d | %s",
                    turn_no, score_before, score_after, peak, peak_decay_score,
                    "✓ 峰值衰减" if peak_decay_score > 0 else
                    ("— 处于峰值" if peak_decay_score == 0 else "✗ 超越峰值"),
                )
        i += 1

    n = len(score_list)
    score_mean = sum(score_list) / n if n else 0.0
    effective = sum(1 for s in score_list if s > 0)
    positive_ratio = effective / n if n else 0.0

    conviction_end = conviction_list[-1]["conviction_after"] if conviction_list else 0
    peak_to_end_decay = peak - conviction_end

    logger.info(
        "[Conviction] peak=%d | end=%d | peak_to_end_decay=%+d | "
        "score_mean=%+.1f | positive_ratio=%.1f%%",
        peak, conviction_end, peak_to_end_decay,
        score_mean, positive_ratio * 100,
    )
    return {
        "conviction_list": conviction_list,
        "score_list": score_list,
        "score_mean": score_mean,
        "positive_ratio": positive_ratio,
        "effective_turns": effective,
        "total_therapist_turns": n,
        "conviction_peak": peak,
        "conviction_end": conviction_end,
        "peak_to_end_decay": peak_to_end_decay,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 模块四：全流程评估入口
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_pipeline(
    transcript_path: str,
    output_path: str | None = None,
) -> tuple[dict, str]:
    """
    读取 run_simulation.py 生成的 transcript JSON，执行完整评估并输出报告。

    Parameters
    ----------
    transcript_path : simulation 输出的 JSON 文件路径
    output_path     : 评估报告输出路径（None 则自动生成至 results/eval/{mode}/）

    Returns
    -------
    (eval_report dict, saved_path str)
    """
    # 加载 transcript
    with open(transcript_path, encoding="utf-8") as f:
        sim_result = json.load(f)

    transcript = sim_result.get("transcript", [])
    meta = sim_result.get("meta", {})
    final_cbt_form = sim_result.get("final_cbt_form", {})

    logger.info(
        "[Eval] Loaded transcript: %d entries | PsyQA #%s | turns=%s",
        len(transcript),
        meta.get("psyqa_index", "?"),
        meta.get("turns_completed", "?"),
    )

    llm = LLMClient.from_role("judge")  # 使用 .env 中的 JUDGE_* 配置

    # ── 模块一：IG-PQA ───────────────────────────────────────────────────────
    logger.info("[Eval] Running IG-PQA evaluation...")
    ig_result = eval_ig_pqa_from_transcript(llm, transcript)

    # ── 模块二：CTRS ─────────────────────────────────────────────────────────
    logger.info("[Eval] Running CTRS evaluation...")
    ctrs_result = eval_ctrs(llm, transcript)

    # ── 模块三：信念确信度衰减 ────────────────────────────────────────────────
    logger.info("[Eval] Running Belief Conviction decay evaluation...")
    conviction_result = eval_conviction_from_transcript(llm, transcript)

    # ── 汇总报告 ─────────────────────────────────────────────────────────────
    report = {
        "meta": meta,
        "final_cbt_form": final_cbt_form,
        "ig_pqa": {
            "ig_mean_bits": ig_result["ig_mean"],
            "ig_positive_ratio": ig_result["ig_positive_ratio"],
            "high_value_turns": ig_result["high_value_turns"],
            "total_therapist_turns": ig_result["total_therapist_turns"],
            "ig_list": ig_result["ig_list"],
        },
        "ctrs": {
            "understanding": ctrs_result["understanding"],
            "guided_discovery": ctrs_result["guided_discovery"],
            "interpersonal_effectiveness": ctrs_result["interpersonal_effectiveness"],
            "ctrs_avg": ctrs_result["ctrs_avg"],
            "justification": ctrs_result["justification"],
        },
        "belief_conviction": {
            "conviction_peak": conviction_result["conviction_peak"],
            "conviction_end": conviction_result["conviction_end"],
            "peak_to_end_decay": conviction_result["peak_to_end_decay"],
            "score_mean": conviction_result["score_mean"],
            "positive_ratio": conviction_result["positive_ratio"],
            "effective_turns": conviction_result["effective_turns"],
            "total_therapist_turns": conviction_result["total_therapist_turns"],
            "score_list": conviction_result["score_list"],
            "conviction_list": conviction_result["conviction_list"],
        },
    }

    # 打印摘要
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  CBT-Discover 评估报告  |  PsyQA #{meta.get('psyqa_index', '?')}")
    print(sep)
    print(f"  对话轮数         : {meta.get('turns_completed', '?')}")
    print(f"  最终认知扭曲     : {final_cbt_form.get('cognitive_distortion', '未识别')}")
    print(f"  最终情绪         : {final_cbt_form.get('emotion', '未识别')}")
    print(f"\n  【IG-PQA 事实证据清晰度增益评测】")
    print(f"  平均 IG          : {ig_result['ig_mean']:.4f} bits")
    print(f"  高价值提问比例   : {ig_result['ig_positive_ratio']*100:.1f}%  "
          f"({ig_result['high_value_turns']}/{ig_result['total_therapist_turns']} 轮)")
    print(f"\n  【CTRS 临床保真度评分 (0-6)】")
    print(f"  理解与共情       : {ctrs_result['understanding']} / 6")
    print(f"  引导式发现       : {ctrs_result['guided_discovery']} / 6")
    print(f"  人际效能         : {ctrs_result['interpersonal_effectiveness']} / 6")
    print(f"  综合平均         : {ctrs_result['ctrs_avg']:.2f} / 6")
    print(f"\n  判分依据: {ctrs_result['justification'][:200]}")
    print(f"\n  【信念确信度（Belief Conviction）峰值衰减指标】")
    print(f"  历史峰值确信度   : {conviction_result['conviction_peak']} / 100")
    print(f"  最终确信度       : {conviction_result['conviction_end']} / 100")
    print(f"  峰值→末尾衰减    : {conviction_result['peak_to_end_decay']:+d} 分")
    print(f"  每轮平均得分     : {conviction_result['score_mean']:+.1f} 分")
    print(f"  有效松动轮次     : {conviction_result['effective_turns']}/{conviction_result['total_therapist_turns']} 轮  "
          f"({conviction_result['positive_ratio']*100:.1f}%)")
    print(f"{sep}\n")

    # 写入 JSON
    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = meta.get("mode", "unknown")
        idx = meta.get("psyqa_index", 0)
        output_path = f"results/eval/{mode}/psyqa{idx}_{ts}.json"

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("[Eval] Report saved to: %s", output_path)

    return report, output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CBT-Discover 评估管线：IG-PQA + CTRS + Belief Conviction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--transcript", nargs="+", required=True, metavar="FILE",
        help=(
            "一个或多个 transcript JSON 文件路径（由 run_simulation.py 生成）。"
            "也可传入目录路径，将自动遍历其下所有 .json 文件。"
            "示例：--transcript results/sim/cbt-discover/psyqa0.json results/sim/cbt-discover/psyqa1.json"
            "      --transcript results/sim/cbt-discover/"
        ),
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="评估报告输出目录（默认 results/eval/{mode}/），文件名自动生成为 psyqa{index}_{timestamp}.json",
    )
    args = parser.parse_args()

    # 收集所有目标文件
    transcript_files: list[Path] = []
    for item in args.transcript:
        p = Path(item)
        if p.is_dir():
            transcript_files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            transcript_files.append(p)
        else:
            logger.warning("[CLI] 路径不存在，已跳过: %s", item)

    if not transcript_files:
        logger.error("[CLI] 未找到任何有效的 transcript 文件，退出。")
        sys.exit(1)

    print(f"\n共 {len(transcript_files)} 个 transcript 文件待评测：")
    for f in transcript_files:
        print(f"  {f}")

    saved_paths: list[str] = []
    for tf in transcript_files:
        if args.output_dir is not None:
            from datetime import datetime
            # 从文件名推断 psyqa index，保持命名一致
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(Path(args.output_dir) / f"{tf.stem}_eval_{ts}.json")
        else:
            out_path = None  # run_eval_pipeline 自动生成至 results/eval/{mode}/

        _, saved = run_eval_pipeline(
            transcript_path=str(tf),
            output_path=out_path,
        )
        saved_paths.append(saved)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  全部评测完成  |  共 {len(saved_paths)} 份报告")
    for p in saved_paths:
        print(f"    {p}")
    print(f"{sep}\n")

