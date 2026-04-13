"""
run_simulation.py — 沙盘模拟脚本

支持两种运行模式（--mode）：
  cbt-discover  使用 DiagnosticianNode + TherapistNode 双Agent系统
                模型由 .env 中 SUPERVISOR_* 和 THERAPIST_* 分别配置
  baseline      使用单一通用大模型作为咨询师（无后台诊断，用于对比实验）
                模型由 .env 中 BASELINE_* 配置

两种模式输出的 transcript JSON 格式完全相同，可直接送入 eval_pipeline.py 评测。

用法：
    # 单条记录
    python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0

    # 多条记录（空格分隔）
    python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0 1 2

    # 闭区间范围（含两端）
    python run_simulation.py --mode baseline --turns 10 --psyqa-index 0-4

    # 混合写法
    python run_simulation.py --mode cbt-discover --turns 10 --psyqa-index 0-2 5 8-9

    # 指定输出目录（文件名自动生成）
    python run_simulation.py --mode cbt-discover --psyqa-index 0-4 --output-dir results/sim/cbt-discover/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from agents import LLMClient, PatientNode, build_intervention_graph, make_initial_state

# ─────────────────────────────────────────────────────────────────────────────
# 日志：同时输出到控制台和 logs/simulation_<timestamp>.log
# ─────────────────────────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.DEBUG)  # 文件记录 DEBUG 及以上

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(logging.INFO)  # 控制台只显示 INFO 及以上

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[_console_handler, _file_handler],
)
logger = logging.getLogger("cbt.simulation")
logger.info("[Log] Session log file: %s", _LOG_FILE)

SEP  = "─" * 60
SEP2 = "═" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 单模型咨询师（对比实验用）
# ─────────────────────────────────────────────────────────────────────────────

_BASELINE_SYSTEM = (
    "你是一位专业的认知行为治疗（CBT）心理咨询师。"
    "请用温暖、共情的语气与来访者对话，运用CBT技术（苏格拉底提问、认知重构等）"
    "帮助其识别和改变不合理信念。每次回复简洁聚焦，不超过150字。"
)


class BaselineTherapist:
    """
    单一通用大模型咨询师。
    使用 .env 中 BASELINE_* 配置，无 DiagnosticianNode，无 CBT 表单。
    与 CBT-Discover 系统进行公平对比实验时使用。
    """

    def __init__(self):
        self.llm = LLMClient.from_role("baseline")
        self._history: list[dict[str, str]] = []
        logger.info("[Baseline] model=%s  base_url=%s", self.llm.model, self.llm.base_url)

    def reset(self) -> None:
        self._history = []

    def chat(self, patient_msg: str) -> str:
        self._history.append({"role": "user", "content": patient_msg})
        messages = [{"role": "system", "content": _BASELINE_SYSTEM}] + self._history
        try:
            reply = self.llm.chat(messages=messages, temperature=0.7)
        except Exception as exc:
            logger.error("[Baseline] LLM call failed: %s", exc)
            reply = "非常抱歉，我现在无法回应，请稍后再试。"
        self._history.append({"role": "assistant", "content": reply})
        return reply


# ─────────────────────────────────────────────────────────────────────────────
# PsyQA 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_psyqa_entry(psyqa_path: str, index: int) -> dict:
    """从 PsyQA_full.json 读取指定索引的条目，返回完整字段。"""
    with open(psyqa_path, encoding="utf-8") as f:
        data = json.load(f)
    entry = data[index % len(data)]
    return {
        "question":    entry.get("question", ""),
        "description": entry.get("description", ""),
        "keywords":    entry.get("keywords", ""),
        "answers":     entry.get("answers", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CBT-Discover 模式主循环
# ─────────────────────────────────────────────────────────────────────────────

def _run_cbt_discover(turns: int, psyqa_entry: dict, background: str) -> tuple:
    """
    双Agent模式：
      - DiagnosticianNode 使用 SUPERVISOR_* 模型（后台高推理）
      - TherapistNode     使用 THERAPIST_*  模型（前端对话）
    节点各自从 .env 读取配置，无需手动传参。
    Returns: (transcript, final_cbt_form, turns_completed)
    """
    intervention_graph = build_intervention_graph()  # 自动从 .env 读取各角色模型
    patient_node = PatientNode(background=background)

    opening = psyqa_entry["question"] or "老师，我最近真的很绝望，不知道该怎么办了。"
    state = make_initial_state(patient_opening=opening)
    transcript: list[dict] = [{"role": "patient", "content": opening, "turn": 0}]

    print(f"[来访者 开场]\n{opening}\n{SEP}")

    turn = 0
    while turn < turns:
        turn += 1
        logger.info("[CBT-Discover] ── Turn %d/%d ──", turn, turns)

        try:
            state = intervention_graph.invoke(state)
        except Exception as exc:
            logger.error("[CBT-Discover] Intervention graph failed at turn %d: %s", turn, exc)
            break

        therapist_reply = state.get("last_therapist_response", "")
        inner_monologue = state.get("last_inner_monologue", "")
        cbt_form        = state.get("cbt_form", {})

        print(f"[咨询师 Turn {turn}]\n{therapist_reply}")
        logger.info("[CBT-Discover] distortion=%s | emotion=%s",
                    cbt_form.get("cognitive_distortion"), cbt_form.get("emotion"))

        transcript.append({
            "role": "therapist",
            "content": therapist_reply,
            "inner_monologue": inner_monologue,
            "cbt_form_snapshot": dict(cbt_form),
            "turn": turn,
        })

        try:
            patient_updates = patient_node(state)
            state.update(patient_updates)
        except Exception as exc:
            logger.error("[CBT-Discover] Patient node failed at turn %d: %s", turn, exc)
            break

        patient_reply = state.get("last_patient_msg", "")
        print(f"{SEP}\n[来访者 Turn {turn}]\n{patient_reply}\n{SEP}")
        transcript.append({"role": "patient", "content": patient_reply, "turn": turn})

    return transcript, dict(state.get("cbt_form", {})), turn


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 模式主循环
# ─────────────────────────────────────────────────────────────────────────────

def _run_baseline(turns: int, psyqa_entry: dict, background: str) -> tuple:
    """
    单模型对比模式：BaselineTherapist (BASELINE_*) 直接与患者对话。
    无 CBT 表单，无诊断节点。
    Returns: (transcript, final_cbt_form, turns_completed)
    """
    therapist    = BaselineTherapist()
    patient_node = PatientNode(background=background)

    opening = psyqa_entry["question"] or "老师，我最近真的很绝望，不知道该怎么办了。"

    # PatientNode 需要 DialogueState 字段，构造一个简单 dict 兼容
    state: dict = {
        "chat_history": [{"role": "user", "content": opening}],
        "last_therapist_response": "",
        "last_patient_msg": opening,
        "last_inner_monologue": "",
        "cbt_form": {},
        "entropy_scores": [],
        "turn_count": 0,
    }
    transcript: list[dict] = [{"role": "patient", "content": opening, "turn": 0}]
    current_patient_msg = opening

    print(f"[来访者 开场]\n{opening}\n{SEP}")

    turn = 0
    while turn < turns:
        turn += 1
        logger.info("[Baseline] ── Turn %d/%d ──", turn, turns)

        try:
            therapist_reply = therapist.chat(current_patient_msg)
        except Exception as exc:
            logger.error("[Baseline] Therapist failed at turn %d: %s", turn, exc)
            break

        print(f"[咨询师(Baseline) Turn {turn}]\n{therapist_reply}")
        state["last_therapist_response"] = therapist_reply
        state["turn_count"] = turn
        state["chat_history"] = state["chat_history"] + [
            {"role": "assistant", "content": therapist_reply}
        ]
        transcript.append({"role": "therapist", "content": therapist_reply, "turn": turn})

        try:
            patient_updates = patient_node(state)
            state.update(patient_updates)
        except Exception as exc:
            logger.error("[Baseline] Patient node failed at turn %d: %s", turn, exc)
            break

        current_patient_msg = state.get("last_patient_msg", "")
        print(f"{SEP}\n[来访者 Turn {turn}]\n{current_patient_msg}\n{SEP}")
        transcript.append({"role": "patient", "content": current_patient_msg, "turn": turn})

    return transcript, {}, turn


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    mode: str = "cbt-discover",
    turns: int = 10,
    psyqa_path: str = "datasets/PsyQA/PsyQA_full.json",
    psyqa_index: int = 0,
    output_path: str | None = None,
) -> tuple[list[dict], str]:
    """
    执行单条 PsyQA 记录的沙盘模拟并保存 transcript。

    Parameters
    ----------
    mode        : "cbt-discover" — 双Agent系统（SUPERVISOR_* + THERAPIST_*）
                  "baseline" — 单一通用大模型（BASELINE_*），用于对比实验
    turns       : 最大对话轮数
    psyqa_path  : PsyQA 数据集路径
    psyqa_index : 使用第几条 PsyQA 记录作为患者背景
    output_path : 结果 JSON 输出路径（None 则自动生成至 results/sim/{mode}/）

    Returns
    -------
    (transcript, saved_path)
    """
    # ── 加载患者背景 ──────────────────────────────────────────────────────────
    psyqa_entry = {"question": "", "description": ""}
    if Path(psyqa_path).exists():
        psyqa_entry = load_psyqa_entry(psyqa_path, psyqa_index)
        logger.info("[Sim] PsyQA #%d: %s", psyqa_index, psyqa_entry["question"][:60])
    else:
        logger.warning("[Sim] PsyQA not found: %s — using default background", psyqa_path)

    background = ""
    if psyqa_entry["question"]:
        parts = []
        parts.append("【来访者主诉标题】\n" + psyqa_entry["question"])
        if psyqa_entry["description"]:
            parts.append("【详细描述（来访者原始陈述）】\n" + psyqa_entry["description"])
        if psyqa_entry["keywords"]:
            parts.append("【话题标签】" + psyqa_entry["keywords"])
        # 取第一条参考答案（若存在）帮助模型理解咨询方向，不暴露给来访者角色
        ref_answers = psyqa_entry.get("answers", [])
        if ref_answers:
            first_answer = ref_answers[0].get("answer_text", "") if isinstance(ref_answers[0], dict) else str(ref_answers[0])
            if first_answer:
                parts.append("【参考咨询师视角（仅用于推断来访者深层情绪，不要直接引用）】\n" + first_answer[:500])
        background = "\n\n".join(parts)

    print(f"\n{SEP2}")
    print(f"  CBT-Discover 沙盘模拟  |  mode={mode}  |  PsyQA #{psyqa_index}  |  最大 {turns} 轮")
    print(SEP2)

    # ── 执行对应模式 ──────────────────────────────────────────────────────────
    if mode == "cbt-discover":
        transcript, final_cbt_form, turns_done = _run_cbt_discover(turns, psyqa_entry, background)
    elif mode == "baseline":
        transcript, final_cbt_form, turns_done = _run_baseline(turns, psyqa_entry, background)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Choose 'cbt-discover' or 'baseline'.")

    print(f"\n{SEP2}")
    print(f"  模拟结束  |  mode={mode}  |  共完成 {turns_done} 轮对话")
    print(f"{SEP2}\n")

    # ── 保存结果 ──────────────────────────────────────────────────────────────
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/sim/{mode}/psyqa{psyqa_index}_{ts}.json"

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "meta": {
            "mode": mode,
            "psyqa_index": psyqa_index,
            "psyqa_question": psyqa_entry.get("question", ""),
            "turns_completed": turns_done,
            "max_turns": turns,
            "timestamp": datetime.now().isoformat(),
        },
        "final_cbt_form": final_cbt_form,
        "transcript": transcript,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("[Sim] Transcript saved to: %s", output_path)
    return transcript, output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_indices(index_args: list[str]) -> list[int]:
    """
    将 CLI 传入的索引参数解析为整数列表。
    支持两种格式（可混用）：
      单个索引  : "0" "3" "7"
      闭区间范围: "0-4"  => [0, 1, 2, 3, 4]
    示例："0-2 5 8-9" => [0, 1, 2, 5, 8, 9]
    """
    indices: list[int] = []
    for token in index_args:
        if "-" in token:
            parts = token.split("-", 1)
            start, end = int(parts[0]), int(parts[1])
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    # 去重并保序
    seen: set[int] = set()
    result: list[int] = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CBT-Discover 沙盘模拟：双Agent系统 vs 单模型基线对比",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["cbt-discover", "baseline"], default="cbt-discover",
        help="运行模式：cbt-discover=双Agent系统(THERAPIST+SUPERVISOR)，baseline=单一大模型(BASELINE)",
    )
    parser.add_argument("--turns", type=int, default=10, help="最大对话轮数")
    parser.add_argument("--psyqa", default="datasets/PsyQA/PsyQA_full.json",
                        help="PsyQA 数据集路径")
    parser.add_argument(
        "--psyqa-index", nargs="+", default=["0"], metavar="IDX",
        help=(
            "要测试的 PsyQA 条目索引，支持多值和闭区间范围，可混用。"
            "示例：--psyqa-index 0 1 2   或   --psyqa-index 0-4   或   --psyqa-index 0-2 5 8-9"
        ),
    )
    parser.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="输出目录（默认 results/sim/{mode}/），文件名自动生成为 psyqa{index}_{timestamp}.json",
    )
    args = parser.parse_args()

    indices = _parse_indices(args.psyqa_index)
    saved_paths: list[str] = []

    print(f"\n共 {len(indices)} 条 PsyQA 记录待模拟：{indices}")

    for idx in indices:
        if args.output_dir is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(Path(args.output_dir) / f"psyqa{idx}_{ts}.json")
        else:
            out_path = None  # run_simulation 自动生成至 results/sim/{mode}/

        _, saved = run_simulation(
        mode=args.mode,
        turns=args.turns,
        psyqa_path=args.psyqa,
            psyqa_index=idx,
            output_path=out_path,
    )
        saved_paths.append(saved)

    print(f"\n{'═'*60}")
    print(f"  全部模拟完成  |  共 {len(saved_paths)} 条记录")
    for p in saved_paths:
        print(f"    {p}")
    print(f"{'═'*60}\n")
