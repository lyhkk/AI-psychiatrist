import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from scipy.stats import entropy as scipy_entropy
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv(Path(__file__).parent / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_logging(log_dir: str = "logs") -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"eval_{timestamp}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    class TqdmHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                tqdm.write(self.format(record), file=sys.stderr)
                self.flush()
            except Exception:
                self.handleError(record)

    ch = TqdmHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    log = logging.getLogger("cbt.eval")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False
    log.info("Log file: %s", log_file.resolve())
    return log


logger = _setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    elif p.suffix == ".tsv":
        import csv
        rows = []
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
        return rows
    raise ValueError(f"Unsupported dataset format: {p.suffix}")


# ═══════════════════════════════════════════════════════════════════════════════
# Cache helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cache_path(cache_dir: str, task: str) -> Path:
    return Path(cache_dir) / f"{task}.jsonl"


def load_cache(cache_dir: str, task: str) -> dict[str, str]:
    p = cache_path(cache_dir, task)
    if not p.exists():
        return {}
    cache: dict[str, str] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cache[obj["input"]] = obj["output"]
            except (json.JSONDecodeError, KeyError):
                pass
    logger.info("[cache] Loaded %d entries from %s", len(cache), p)
    return cache


def append_cache(cache_dir: str, task: str, input_text: str, output_text: str) -> None:
    p = cache_path(cache_dir, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"input": input_text, "output": output_text}, ensure_ascii=False) + "\n")


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Model interface
# ═══════════════════════════════════════════════════════════════════════════════

class BaseModel:
    name: str = "base"

    def reset(self) -> None:
        raise NotImplementedError

    def chat(self, user_message: str) -> str:
        raise NotImplementedError


class CBTerModel(BaseModel):
    """
    Wrapper around the CBT-er Flask backend.
    POST /chat  {"message": "..."}  ->  {"reply": "..."}
    POST /reset                      ->  {"ok": true}
    """
    name = "cbter"

    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http = requests.Session()

    def reset(self) -> None:
        resp = self._http.post(f"{self.base_url}/reset", timeout=self.timeout)
        resp.raise_for_status()
        logger.debug("[CBTerModel] Session reset: %s", resp.json())

    def chat(self, user_message: str) -> str:
        resp = self._http.post(
            f"{self.base_url}/chat",
            json={"message": user_message},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("reply", "")


class BaselineModel(BaseModel):
    name = "baseline"
    _SYSTEM_PROMPT = (
        "你是一位专业的认知行为治疗（CBT）心理咨询师。"
        "请用温暖、共情的语气与来访者对话，运用CBT技术（苏格拉底提问、认知重构等）帮助其识别和改变不合理信念。"
        "每次回复简洁聚焦，不超过150字。"
    )

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "", timeout: int = 120):
        self.api_key  = api_key  or os.getenv("BASELINE_API_KEY", "")
        self.base_url = (base_url or os.getenv("BASELINE_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model    = model    or os.getenv("BASELINE_MODEL", "gpt-4o-mini")
        self.timeout  = timeout
        self._history: list[dict] = []

    def reset(self) -> None:
        self._history = []
        logger.debug("[BaselineModel] History cleared")

    def chat(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self._SYSTEM_PROMPT}] + self._history
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": 0.7}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        self._history.append({"role": "assistant", "content": reply})
        logger.debug("[BaselineModel] reply: %s", reply[:80])
        return reply


__MODEL__ = CBTerModel


# ═══════════════════════════════════════════════════════════════════════════════
# Judge (LLM-as-a-Judge)
# ═══════════════════════════════════════════════════════════════════════════════

class Judge:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: int = 120):
        self.api_key  = api_key
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout

    def judge(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers, json=payload, timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# Basic CTRS metric
# ═══════════════════════════════════════════════════════════════════════════════

def score_ctrs(judge: Judge, response: str, context: str) -> dict:
    CTRS_PROMPT = """\
You are an expert clinical psychologist trained in Cognitive Behavioral Therapy (CBT).
Evaluate the therapist response using CTRS. Score each dimension 0-6:
1. understanding
2. interpersonal_effectiveness
3. collaboration
4. guided_discovery

Context:
{context}

Therapist response:
{response}

Return ONLY a JSON object with those 4 keys and integer scores 0-6."""
    raw = judge.judge(
        system_prompt="You are a clinical evaluation assistant.",
        user_prompt=CTRS_PROMPT.format(context=context, response=response),
    )
    result = extract_json(raw)
    return result if result is not None else {}


# ═══════════════════════════════════════════════════════════════════════════════
# 改进二：基于信息熵 (Information Gain) 的 PQA 评测
# ═══════════════════════════════════════════════════════════════════════════════

_DISTORTION_LABELS_ZH = [
    "非此即彼", "以偏概全", "心理过滤", "否定正面思考",
    "读心术", "先知错误", "放大", "情绪化推理",
    "应该式", "乱贴标签", "罪责归己", "罪责归他",
]

_ENTROPY_JUDGE_SYSTEM = """\
你是一个临床CBT督导。根据提供的对话历史，列出当前来访者最有可能存在的认知扭曲类型，
并以JSON格式为每种类型分配概率权重（总和为1.0）。
只从以下列表中选择：非此即彼、以偏概全、心理过滤、否定正面思考、读心术、先知错误、
放大、情绪化推理、应该式、乱贴标签、罪责归己、罪责归他。
只输出3种可能性最高的类型及其概率，其余合并为"其他"。
输出格式示例：{"非此即彼": 0.7, "以偏概全": 0.2, "读心术": 0.1}
只输出JSON对象，不要有任何额外文字。
"""


def _compute_shannon_entropy(prob_dict: dict) -> float:
    probs = [float(v) for v in prob_dict.values()]
    if not probs:
        return 0.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]
    if _SCIPY_AVAILABLE:
        return float(scipy_entropy(probs, base=2))
    return float(-sum(p * math.log2(p) for p in probs if p > 0))


def _get_distortion_probs(judge: Judge, conversation_text: str) -> dict[str, float]:
    try:
        raw = judge.judge(
            system_prompt=_ENTROPY_JUDGE_SYSTEM,
            user_prompt=f"以下是对话记录：\n\n{conversation_text}\n\n请输出概率JSON。",
        )
        result = extract_json(raw)
        if result and isinstance(result, dict):
            clean = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            if clean:
                return clean
    except Exception as exc:
        logger.warning("[IG-PQA] 概率分布获取失败: %s", exc)
    n = len(_DISTORTION_LABELS_ZH)
    return {lbl: 1.0 / n for lbl in _DISTORTION_LABELS_ZH}


def compute_information_gain(judge: Judge, conv_before: str, conv_after: str) -> float:
    """
    IG = H(P_before) - H(P_after)
    H: Shannon entropy in bits.
    Positive IG means the therapist question successfully reduced uncertainty.
    """
    h_before = _compute_shannon_entropy(_get_distortion_probs(judge, conv_before))
    h_after  = _compute_shannon_entropy(_get_distortion_probs(judge, conv_after))
    ig = h_before - h_after
    logger.debug("[IG-PQA] H_before=%.4f  H_after=%.4f  IG=%.4f", h_before, h_after, ig)
    return ig


def eval_pqa_information_gain(judge: Judge, generated_replies: list[dict]) -> dict:
    """
    对一批单轮生成结果计算 IG-PQA 指标。

    Parameters
    ----------
    generated_replies : list of {"user_input": str, "reply": str}

    Returns
    -------
    {
        "ig_pqa_mean"           : float,  # 平均 IG（bits）
        "ig_pqa_positive_ratio" : float,  # IG > 0 的比例
        "ig_list"               : list[float],
    }
    """
    ig_list: list[float] = []
    pbar = tqdm(
        enumerate(generated_replies),
        total=len(generated_replies),
        desc="[PQA-IG] entropy judge",
        unit="sample",
        dynamic_ncols=True,
    )
    for idx, item in pbar:
        user_input = item["user_input"]
        reply = item["reply"]
        if not reply:
            ig_list.append(0.0)
            continue
        conv_before = f"来访者：{user_input}"
        conv_after  = f"来访者：{user_input}\n咨询师：{reply}"
        ig = compute_information_gain(judge, conv_before, conv_after)
        ig_list.append(ig)
        pbar.set_postfix({"IG": f"{ig:.3f}"})
        logger.info("[PQA-IG] #%04d | IG=%.4f", idx, ig)

    n = len(ig_list)
    ig_mean = sum(ig_list) / n if n else 0.0
    ig_pos_ratio = sum(1 for v in ig_list if v > 0) / n if n else 0.0
    logger.info(
        "[PQA-IG] mean_IG=%.4f  positive_ratio=%.1f%%  (n=%d)",
        ig_mean, ig_pos_ratio * 100, n,
    )
    return {
        "ig_pqa_mean":           ig_mean,
        "ig_pqa_positive_ratio": ig_pos_ratio,
        "ig_list":               ig_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 改进三：Patient Agent + 多轮沙盘演练 + 全局 CTRS 评测
# ═══════════════════════════════════════════════════════════════════════════════

_PATIENT_SYSTEM_PROMPT = """\
你正在扮演一名寻求心理咨询的来访者。

【你的背景】：你是一名大三学生，因为连续两次面试失败而感到绝望。
【你的认知扭曲】：你存在严重的"灾难化"和"自我贬低"倾向。
  - 你认为两次失败意味着自己永远找不到工作，是个彻底的失败者。
  - 你认为别人都比你优秀，你天生就不如别人。
【你的行为模式（低开放性）】：你非常有防御心。
  - 如果咨询师直接给建议或急于反驳你，你会感到被冒犯，变得沉默或说"我就是这样，没办法"。
  - 只有当咨询师展现出极高的同理心、耐心引导时，你才会逐渐吐露更深的感受。
  - 不要轻易承认自己的想法是错的，即便心里有所触动，也只是稍微松动，不会突然转变。
请根据咨询师的回复自然地进行多轮对话，保持角色一致。用中文回复，语气符合一个沮丧大学生的状态。
"""

_GLOBAL_CTRS_SYSTEM = """\
你是一名资深的CBT临床督导。请阅读以下完整的多轮心理咨询对话记录。
请根据认知疗法评定量表（CTRS）的以下核心维度，对咨询师（Therapist）的整体表现进行0-6分的严格打分
（0=破坏性，1=劣质，2=较差，3=合格，4=良好，5=优秀，6=杰出）：

1. **理解与共情 (understanding)**：咨询师是否准确捕捉到了来访者的内部现实，
   而没有过度迎合（Sycophancy）其病态信念？

2. **引导式发现 (guided_discovery)**：咨询师是否有效使用了苏格拉底式提问，
   引导来访者产生认知顿悟（Cognitive Shift），而不是直接说教？

3. **人际效能 (interpersonal_effectiveness)**：咨询师是否展现了真实的温暖和专业性，
   并成功化解了来访者的防御？

请输出严谨的JSON格式评分，并在 justification 字段中引用对话中的具体句子作为扣分/加分依据。

输出格式：
{
  "understanding": <0-6>,
  "guided_discovery": <0-6>,
  "interpersonal_effectiveness": <0-6>,
  "justification": "<引用原文的具体说明>"
}
只输出JSON，不要任何额外文字。
"""


class PatientAgent:
    """
    改进三：Patient Agent（模拟来访者）。
    维护自己的对话历史，接收治疗师回复后给出来访者的下一句话。
    使用与 Supervisor 相同的高推理模型确保角色扮演质量。
    """

    def __init__(self, judge: Judge):
        # 复用 Judge 的 API 配置，但可通过环境变量单独覆盖
        patient_api_key  = os.getenv("PATIENT_API_KEY",  judge.api_key)
        patient_base_url = os.getenv("PATIENT_BASE_URL", judge.base_url)
        patient_model    = os.getenv("PATIENT_MODEL",    judge.model)
        self._api_key  = patient_api_key
        self._base_url = patient_base_url.rstrip("/")
        self._model    = patient_model
        self._timeout  = judge.timeout
        self._history: list[dict] = []

    def reset(self) -> None:
        self._history = []

    def reply(self, therapist_message: str) -> str:
        """给出来访者对治疗师回复的自然回应。"""
        self._history.append({"role": "user", "content": therapist_message})
        messages = [
            {"role": "system", "content": _PATIENT_SYSTEM_PROMPT},
        ] + self._history
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {"model": self._model, "messages": messages, "temperature": 0.85}
        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers=headers, json=payload, timeout=self._timeout,
        )
        resp.raise_for_status()
        patient_reply = resp.json()["choices"][0]["message"]["content"]
        self._history.append({"role": "assistant", "content": patient_reply})
        return patient_reply


def run_sandbox_session(
    therapist_model: BaseModel,
    patient_agent: PatientAgent,
    opening_statement: str,
    max_turns: int = 15,
) -> list[dict]:
    """
    运行一次 Therapist ↔ Patient 多轮沙盘演练。

    Parameters
    ----------
    therapist_model  : BaseModel 实例（CBTerModel 或 BaselineModel）
    patient_agent    : PatientAgent 实例
    opening_statement: 来访者的开场陈述（第一句话）
    max_turns        : 最大对话轮数（每轮 = 治疗师1条 + 来访者1条）

    Returns
    -------
    list of {"role": "therapist"|"patient", "content": str}
    """
    therapist_model.reset()
    patient_agent.reset()
    transcript: list[dict] = []

    # 来访者先开口
    current_patient_msg = opening_statement
    transcript.append({"role": "patient", "content": current_patient_msg})
    logger.info("[Sandbox] 来访者(开场): %s", current_patient_msg[:80])

    for turn in range(max_turns):
        # ── 治疗师回复 ──────────────────────────────────────────────────────
        try:
            therapist_reply = therapist_model.chat(current_patient_msg)
        except Exception as exc:
            logger.warning("[Sandbox] turn %d 治疗师调用失败: %s", turn, exc)
            break
        transcript.append({"role": "therapist", "content": therapist_reply})
        logger.info("[Sandbox] turn %02d | 治疗师: %s", turn + 1, therapist_reply[:80])

        # ── 来访者回复 ──────────────────────────────────────────────────────
        try:
            current_patient_msg = patient_agent.reply(therapist_reply)
        except Exception as exc:
            logger.warning("[Sandbox] turn %d 来访者调用失败: %s", turn, exc)
            break
        transcript.append({"role": "patient", "content": current_patient_msg})
        logger.info("[Sandbox] turn %02d | 来访者: %s", turn + 1, current_patient_msg[:80])

    return transcript


def score_global_ctrs(judge: Judge, transcript: list[dict]) -> dict:
    """
    将完整对话记录送入强力裁判模型，进行全局 CTRS 评分（改进三核心）。

    Returns
    -------
    dict 包含 understanding / guided_discovery / interpersonal_effectiveness / justification
    """
    # 格式化对话记录
    lines = []
    for entry in transcript:
        role = "咨询师" if entry["role"] == "therapist" else "来访者"
        lines.append(f"{role}：{entry['content']}")
    transcript_text = "\n".join(lines)

    user_prompt = f"以下是完整的 {len(transcript)} 条对话记录：\n\n{transcript_text}\n\n请按要求评分。"
    try:
        raw = judge.judge(
            system_prompt=_GLOBAL_CTRS_SYSTEM,
            user_prompt=user_prompt,
        )
        # extract_json 只取第一个 {...}，justification 内若有嵌套花括号会截断
        # 用更宽松的正则匹配最外层完整 JSON
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            result = json.loads(match.group())
            return result
    except Exception as exc:
        logger.warning("[GlobalCTRS] 评分失败: %s", exc)
    return {}


def eval_sandbox(
    therapist_model: BaseModel,
    judge: Judge,
    num_sessions: int = 3,
    max_turns: int = 15,
    opening_statement: str = "老师，我最近真的很绝望。我连续两次面试都失败了，我觉得我这辈子可能真的找不到工作了。",
) -> dict:
    """
    改进三主入口：运行多次沙盘演练并汇总全局 CTRS 结果。

    Parameters
    ----------
    num_sessions     : 运行几次独立会话（取平均，降低随机性）
    max_turns        : 每次会话最大轮数
    opening_statement: 来访者固定开场白

    Returns
    -------
    {
        "sandbox_ctrs_understanding"             : float,
        "sandbox_ctrs_guided_discovery"          : float,
        "sandbox_ctrs_interpersonal_effectiveness": float,
        "sandbox_ctrs_avg"                       : float,
        "sandbox_transcripts"                    : list[list[dict]],
    }
    """
    logger.info("=" * 60)
    logger.info("TASK: Sandbox Role-Play  (%d sessions × %d turns)  [model=%s]",
                num_sessions, max_turns, therapist_model.name)
    logger.info("=" * 60)

    patient = PatientAgent(judge)
    keys = ["understanding", "guided_discovery", "interpersonal_effectiveness"]
    accum = {k: 0.0 for k in keys}
    valid = 0
    transcripts = []

    for session_idx in range(num_sessions):
        logger.info("[Sandbox] ── Session %d/%d ──", session_idx + 1, num_sessions)
        transcript = run_sandbox_session(
            therapist_model, patient, opening_statement, max_turns=max_turns
        )
        transcripts.append(transcript)
        scores = score_global_ctrs(judge, transcript)
        if scores:
            for k in keys:
                accum[k] += float(scores.get(k, 0))
            valid += 1
            avg_s = sum(scores.get(k, 0) for k in keys) / len(keys)
            logger.info(
                "[Sandbox] Session %d | avg=%.2f | und=%.0f gui=%.0f int=%.0f | %s",
                session_idx + 1, avg_s,
                scores.get("understanding", 0),
                scores.get("guided_discovery", 0),
                scores.get("interpersonal_effectiveness", 0),
                str(scores.get("justification", ""))[:120],
            )
        else:
            logger.warning("[Sandbox] Session %d 评分失败，跳过", session_idx + 1)

    if valid > 0:
        avg_scores = {k: accum[k] / valid for k in keys}
    else:
        avg_scores = {k: 0.0 for k in keys}

    sandbox_avg = sum(avg_scores.values()) / len(keys)

    logger.info("-" * 60)
    logger.info("[Sandbox] CTRS Results (valid=%d/%d):", valid, num_sessions)
    for k, v in avg_scores.items():
        logger.info("  %-40s %.2f / 6", k, v)
    logger.info("  %-40s %.2f / 6", "[SANDBOX OVERALL AVG]", sandbox_avg)
    logger.info("-" * 60)

    return {
        "sandbox_ctrs_understanding":              avg_scores["understanding"],
        "sandbox_ctrs_guided_discovery":           avg_scores["guided_discovery"],
        "sandbox_ctrs_interpersonal_effectiveness": avg_scores["interpersonal_effectiveness"],
        "sandbox_ctrs_avg":                        sandbox_avg,
        "sandbox_transcripts":                     transcripts,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset labels
# ═══════════════════════════════════════════════════════════════════════════════

DISTORTION_LABELS = [
    "非此即彼", "以偏概全", "心理过滤", "否定正面思考", "读心术",
    "先知错误", "放大", "情绪化推理", "应该式", "乱贴标签", "罪责归己", "罪责归他",
]

CBT_BENCH_LABELS = [
    "I am powerless, weak, vulnerable", "I am needy", "I am out of control",
    "I am helpless", "I am a failure, loser", "I am a victim",
    "I am bound to be rejected", "I am bound to be alone", "I am unlovable",
    "I am bound to be abandoned", "I am worthless, waste", "I am immoral",
    "I am bad - dangerous, toxic, evil", "I am trapped", "I am defective",
    "I don't deserve to live", "I am incompetent",
    "I am unattractive", "I am undesirable, unwanted",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class LocalCBTEvaluator:
    """End-to-end evaluation harness for the CBT-er system."""

    def __init__(self, model: BaseModel, judge: Judge):
        self.model = model
        self.judge = judge
        self.mlb_social = MultiLabelBinarizer(classes=DISTORTION_LABELS)
        self.mlb_social.fit([DISTORTION_LABELS])
        self.mlb_cbt = MultiLabelBinarizer(classes=CBT_BENCH_LABELS)
        self.mlb_cbt.fit([CBT_BENCH_LABELS])

    # ── Data loading ─────────────────────────────────────────────────────────

    @staticmethod
    def _subsample(df: pd.DataFrame, max_samples: int | None) -> pd.DataFrame:
        if max_samples is not None:
            twenty_pct = max(max_samples, int(len(df) * 0.2))
            sampled = df.sample(n=min(twenty_pct, len(df)), random_state=42)
            return sampled.head(max_samples).reset_index(drop=True)
        return df.sample(frac=0.2, random_state=42).reset_index(drop=True)

    def load_local_socialcd(self, file_path: str, max_samples: int | None = None) -> pd.DataFrame:
        logger.info("Loading SocialCD-3K from: %s", file_path)
        df = pd.read_csv(file_path, sep="\t")
        test_df = self._subsample(df, max_samples)
        logger.info("SocialCD-3K: %d rows selected (total=%d, max_samples=%s)",
                    len(test_df), len(df), max_samples)
        processed = []
        for _, row in test_df.iterrows():
            ground_truth = [lbl for lbl in DISTORTION_LABELS if row.get(lbl, 0) == 1]
            processed.append({"text": row["内容"], "ground_truth": ground_truth})
        return pd.DataFrame(processed)

    def load_local_psyqa(self, file_path: str, max_samples: int | None = None) -> pd.DataFrame:
        logger.info("Loading PsyQA from: %s", file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        test_df = self._subsample(df, max_samples)
        logger.info("PsyQA: %d rows selected (total=%d, max_samples=%s)",
                    len(test_df), len(df), max_samples)
        processed = []
        for _, row in test_df.iterrows():
            user_input = f"标题：{row.get('question', '')}\n描述：{row.get('description', '')}"
            processed.append({"user_input": user_input})
        return pd.DataFrame(processed)

    def load_local_cbt_bench(self, file_path: str, max_samples: int | None = None) -> pd.DataFrame:
        logger.info("Loading CBT-Bench from: %s", file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        test_df = self._subsample(df, max_samples)
        logger.info("CBT-Bench: %d rows selected (total=%d, max_samples=%s)",
                    len(test_df), len(df), max_samples)
        processed = []
        for _, row in test_df.iterrows():
            context = f"Situation: {row.get('situation', '')}\nThoughts: {row.get('thoughts', '')}"
            processed.append({"text": context, "ground_truth": row.get("core_belief_fine_grained", "")})
        return pd.DataFrame(processed)

    # ── Evaluation tasks ─────────────────────────────────────────────────────

    def eval_classification_f1(
        self,
        df: pd.DataFrame,
        task_name: str,
        label_list: list[str],
        mlb: MultiLabelBinarizer,
        cache_dir: str = "cache",
        skip_generate: bool = False,
    ) -> float:
        """Evaluate multi-label classification Macro-F1 (SocialCD-3K and CBT-Bench)."""
        logger.info("=" * 60)
        logger.info("TASK: %s Classification  (%d samples)  [model=%s]",
                    task_name, len(df), self.model.name)
        logger.info("=" * 60)

        cache_task = f"{self.model.name}_{task_name}"
        cache = load_cache(cache_dir, cache_task)
        predictions: list[list[str]] = []
        cache_hits = 0

        pbar = tqdm(df.iterrows(), total=len(df), desc=f"[{task_name}] classify",
                    unit="sample", dynamic_ncols=True)
        for idx, row in pbar:
            prompt = (
                f"你是一个CBT心理督导。请分析以下文本存在的认知扭曲/核心信念类型。\n"
                f"严格从以下列表中选择：{label_list}。\n"
                f"必须且只能输出一个JSON格式的字符串数组，不要包含任何其他字符。\n\n"
                f"文本：{row['text']}"
            )
            if prompt in cache:
                response = cache[prompt]
                cache_hits += 1
                pbar.set_postfix({"source": "cache", "hits": cache_hits})
            elif skip_generate:
                raise RuntimeError(
                    f"[{task_name}] sample {idx}: not in cache and "
                    "--skip-generate is set."
                )
            else:
                try:
                    self.model.reset()
                    response = self.model.chat(prompt)
                    append_cache(cache_dir, cache_task, prompt, response)
                except Exception as exc:
                    logger.warning("[%s] sample %d – generation failed: %s", task_name, idx, exc)
                    response = ""
                pbar.set_postfix({"source": "model", "hits": cache_hits})

            raw_response = response
            try:
                clean = re.sub(r"```json\s*|\s*```", "", response).strip()
                pred_list = json.loads(clean)
                valid_preds = [lbl for lbl in label_list if lbl in pred_list]
            except Exception as exc:
                logger.warning("[%s] sample %d – parse failed: %s", task_name, idx, exc)
                valid_preds = []

            predictions.append(valid_preds)
            gt = row["ground_truth"]
            gt_display = gt if isinstance(gt, list) else [gt]

            logger.info("[%s] #%04d | src=%-5s | TEXT: %-40s",
                        task_name, idx,
                        "cache" if prompt in cache else "model",
                        str(row["text"])[:40].replace("\n", " "))
            logger.info("[%s] #%04d | RAW : %s", task_name, idx,
                        raw_response[:120].replace("\n", " "))
            logger.info("[%s] #%04d | GT  : %s", task_name, idx, str(gt_display))
            logger.info("[%s] #%04d | PRED: %s  %s", task_name, idx,
                        str(valid_preds),
                        "✓" if set(valid_preds) & set(gt_display) else "✗")

        logger.info("[%s] Cache hits: %d / %d", task_name, cache_hits, len(df))

        gt_list = df["ground_truth"].tolist()
        gt_wrapped = [[g] if isinstance(g, str) else g for g in gt_list]
        y_true = mlb.transform(gt_wrapped)
        y_pred = mlb.transform(predictions)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        logger.info("-" * 60)
        logger.info("[%s] Macro-F1 Score: %.4f", task_name, macro_f1)
        logger.info("-" * 60)
        return macro_f1

    def eval_therapeutic_generation(
        self,
        df: pd.DataFrame,
        cache_dir: str = "cache",
        skip_generate: bool = False,
        use_ig_pqa: bool = True,
    ) -> tuple:
        """
        Evaluate therapeutic response generation.

        Metrics
        -------
        - CTRS (4 subscores averaged, LLM-as-a-Judge)
        - PQA  – 改进二：基于信息熵的 IG-PQA（use_ig_pqa=True）
                  或旧版正则 PQA（use_ig_pqa=False）
        """
        TASK = "PsyQA"
        logger.info("=" * 60)
        logger.info("TASK: Therapeutic Response Generation  (%d samples)  [model=%s]",
                    len(df), self.model.name)
        logger.info("=" * 60)

        cache_task = f"{self.model.name}_{TASK}"
        gen_cache = load_cache(cache_dir, cache_task)
        generated_replies: list[dict] = []
        cache_hits = 0

        # ── Step 1: collect replies ───────────────────────────────────────────
        pbar = tqdm(df.iterrows(), total=len(df), desc="[PsyQA] generate",
                    unit="sample", dynamic_ncols=True)
        for idx, row in pbar:
            user_input = row["user_input"]
            if user_input in gen_cache:
                reply = gen_cache[user_input]
                cache_hits += 1
                src = "cache"
            elif skip_generate:
                raise RuntimeError(
                    f"[PsyQA] sample {idx}: not in cache and --skip-generate is set."
                )
            else:
                try:
                    self.model.reset()
                    reply = self.model.chat(user_input)
                    append_cache(cache_dir, cache_task, user_input, reply)
                except Exception as exc:
                    logger.warning("[PsyQA] sample %d – generation failed: %s", idx, exc)
                    reply = ""
                src = "model"

            generated_replies.append({"user_input": user_input, "reply": reply})
            logger.info("[PsyQA] #%04d | src=%-5s | INPUT: %s",
                        idx, src, user_input[:80].replace("\n", " "))
            logger.info("[PsyQA] #%04d | REPLY: %s",
                        idx, reply[:120].replace("\n", " "))
            pbar.set_postfix({"source": src, "hits": cache_hits, "reply_len": len(reply)})

        logger.info("[PsyQA] Cache hits: %d / %d", cache_hits, len(df))

        # ── Step 2: PQA ───────────────────────────────────────────────────────
        if use_ig_pqa:
            # 改进二：信息熵 IG-PQA
            ig_result = eval_pqa_information_gain(self.judge, generated_replies)
            pqa_score = ig_result["ig_pqa_positive_ratio"]
            ig_mean   = ig_result["ig_pqa_mean"]
            logger.info("[PsyQA] IG-PQA positive_ratio=%.1f%%  mean_IG=%.4f bits",
                        pqa_score * 100, ig_mean)
        else:
            # 旧版正则 PQA（兼容保留）
            socratic_pat = re.compile(
                r"(\uff1f|\?|你觉得|是什么让你|如果.*?会怎样|有什么证据表明|换个角度)"
            )
            proactive_count = sum(
                1 for item in generated_replies if socratic_pat.search(item["reply"])
            )
            pqa_score = proactive_count / len(generated_replies) if generated_replies else 0.0
            ig_mean = 0.0
            logger.info("[PsyQA] PQA (regex): %.1f%% (%d/%d)",
                        pqa_score * 100, proactive_count, len(generated_replies))

        # ── Step 3: CTRS scoring ──────────────────────────────────────────────
        CTRS_KEYS = ["understanding", "interpersonal_effectiveness",
                     "collaboration", "guided_discovery"]
        ctrs_accum: dict[str, float] = {k: 0.0 for k in CTRS_KEYS}
        valid_evals = 0

        pbar2 = tqdm(enumerate(generated_replies), total=len(generated_replies),
                     desc="[PsyQA] CTRS judge", unit="sample", dynamic_ncols=True)
        for idx, item in pbar2:
            if not item["reply"]:
                continue
            try:
                scores = score_ctrs(self.judge, response=item["reply"],
                                    context=item["user_input"])
                if scores:
                    for k in CTRS_KEYS:
                        ctrs_accum[k] += float(scores.get(k, 0))
                    valid_evals += 1
                    avg_this = sum(scores.get(k, 0) for k in CTRS_KEYS) / len(CTRS_KEYS)
                    logger.info("[CTRS] #%04d | avg=%.2f | %s",
                                idx, avg_this,
                                " ".join(f"{k[:3]}={scores.get(k, 0)}" for k in CTRS_KEYS))
                    pbar2.set_postfix({"avg": f"{avg_this:.2f}"})
            except Exception as exc:
                logger.warning("[CTRS] sample %d – judging failed: %s", idx, exc)

        if valid_evals > 0:
            ctrs_avg = {k: ctrs_accum[k] / valid_evals for k in CTRS_KEYS}
        else:
            ctrs_avg = {k: 0.0 for k in CTRS_KEYS}

        overall_ctrs = sum(ctrs_avg.values()) / len(CTRS_KEYS) if CTRS_KEYS else 0.0

        logger.info("-" * 60)
        logger.info("[PsyQA] CTRS Results (valid=%d/%d):", valid_evals, len(generated_replies))
        for k, v in ctrs_avg.items():
            logger.info("  %-35s %.2f / 6", k, v)
        logger.info("  %-35s %.2f / 6", "[OVERALL AVG]", overall_ctrs)
        if use_ig_pqa:
            logger.info("  %-35s %.4f bits", "IG-PQA Mean", ig_mean)
            logger.info("  %-35s %.1f%%", "IG-PQA Positive Ratio", pqa_score * 100)
        else:
            logger.info("  %-35s %.1f%%", "PQA Ratio (regex)", pqa_score * 100)
        logger.info("-" * 60)

        return overall_ctrs, pqa_score, ctrs_avg, ig_mean


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CBT-er Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Backend / judge ───────────────────────────────────────────────────────
    parser.add_argument("--backend", default="http://127.0.0.1:5000")
    parser.add_argument("--judge-api-key",
                        default=os.getenv("JUDGE_API_KEY", ""))
    parser.add_argument("--judge-base-url",
                        default=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--judge-model",
                        default=os.getenv("JUDGE_MODEL", "gpt-4o"))
    # ── Model type ────────────────────────────────────────────────────────────
    parser.add_argument("--model-type", choices=["cbter", "baseline"], default="cbter")
    parser.add_argument("--baseline-api-key",
                        default=os.getenv("BASELINE_API_KEY", ""))
    parser.add_argument("--baseline-base-url",
                        default=os.getenv("BASELINE_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--baseline-model",
                        default=os.getenv("BASELINE_MODEL", "gpt-4o-mini"))
    # ── Dataset paths ─────────────────────────────────────────────────────────
    parser.add_argument("--socialcd",
                        default=r"datasets/SupervisedVsLLM-EfficacyEval/data/SocialCD-3k/SocialCD-3k.tsv")
    parser.add_argument("--psyqa",
                        default=r"datasets/PsyQA/PsyQA_full.json")
    parser.add_argument("--cbtbench",
                        default=r"datasets/CBT-Bench/core_fine_test.json")
    # ── Cost-control knobs ────────────────────────────────────────────────────
    parser.add_argument("--max-samples", type=int, default=None, metavar="N")
    parser.add_argument("--cache-dir", default="cache", metavar="DIR")
    parser.add_argument("--skip-generate", action="store_true")
    # ── PQA mode ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--pqa-mode", choices=["ig", "regex"], default="ig",
        help="PQA scoring mode: 'ig' = information-gain entropy (改进二), 'regex' = legacy regex.",
    )
    # ── 改进三：沙盘演练 ───────────────────────────────────────────────────────
    parser.add_argument(
        "--sandbox", action="store_true",
        help="Run sandbox role-play evaluation (改进三: Patient Agent + global CTRS).",
    )
    parser.add_argument("--sandbox-sessions", type=int, default=3, metavar="N",
                        help="Number of independent sandbox sessions to average over.")
    parser.add_argument("--sandbox-turns", type=int, default=15, metavar="N",
                        help="Max turns per sandbox session.")
    args = parser.parse_args()

    # ── Instantiate components ────────────────────────────────────────────────
    if args.model_type == "baseline":
        model: BaseModel = BaselineModel(
            api_key=args.baseline_api_key,
            base_url=args.baseline_base_url,
            model=args.baseline_model,
        )
    else:
        model = CBTerModel(base_url=args.backend)

    judge = Judge(
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        model=args.judge_model,
    )
    evaluator = LocalCBTEvaluator(model=model, judge=judge)

    logger.info("model=%s  max_samples=%s  cache_dir=%s  skip_generate=%s  pqa_mode=%s",
                model.name, args.max_samples, args.cache_dir,
                args.skip_generate, args.pqa_mode)

    results: dict = {}

    # ── Task 1: SocialCD-3K ───────────────────────────────────────────────────
    if Path(args.socialcd).exists():
        df_social = evaluator.load_local_socialcd(args.socialcd,
                                                  max_samples=args.max_samples)
        if not df_social.empty:
            f1 = evaluator.eval_classification_f1(
                df_social, "SocialCD-3K", DISTORTION_LABELS, evaluator.mlb_social,
                cache_dir=args.cache_dir, skip_generate=args.skip_generate,
            )
            results["SocialCD-3K Macro-F1"] = f1
    else:
        logger.warning("SocialCD-3K file not found: %s", args.socialcd)

    # ── Task 2: PsyQA (CTRS + IG-PQA) ───────────────────────────────────────
    if Path(args.psyqa).exists():
        df_psyqa = evaluator.load_local_psyqa(args.psyqa,
                                              max_samples=args.max_samples)
        if not df_psyqa.empty:
            use_ig = (args.pqa_mode == "ig")
            overall_ctrs, pqa_score, ctrs_avg, ig_mean = \
                evaluator.eval_therapeutic_generation(
                    df_psyqa,
                    cache_dir=args.cache_dir,
                    skip_generate=args.skip_generate,
                    use_ig_pqa=use_ig,
                )
            results["CTRS Overall Avg"] = overall_ctrs
            if use_ig:
                results["IG-PQA Mean (bits)"] = ig_mean
                results["IG-PQA Positive Ratio"] = pqa_score
            else:
                results["PQA Ratio"] = pqa_score
            results.update({f"CTRS/{k}": v for k, v in ctrs_avg.items()})
    else:
        logger.warning("PsyQA file not found: %s", args.psyqa)

    # ── Task 3: CBT-Bench ─────────────────────────────────────────────────────
    if Path(args.cbtbench).exists():
        df_cbt = evaluator.load_local_cbt_bench(args.cbtbench,
                                                max_samples=args.max_samples)
        if not df_cbt.empty:
            f1_cbt = evaluator.eval_classification_f1(
                df_cbt, "CBT-Bench", CBT_BENCH_LABELS, evaluator.mlb_cbt,
                cache_dir=args.cache_dir, skip_generate=args.skip_generate,
            )
            results["CBT-Bench Macro-F1"] = f1_cbt
    else:
        logger.warning("CBT-Bench file not found: %s", args.cbtbench)

    # ── Task 4 (改进三): Sandbox role-play ────────────────────────────────────
    if args.sandbox:
        sandbox_result = eval_sandbox(
            therapist_model=model,
            judge=judge,
            num_sessions=args.sandbox_sessions,
            max_turns=args.sandbox_turns,
        )
        results["Sandbox CTRS / understanding"]               = sandbox_result["sandbox_ctrs_understanding"]
        results["Sandbox CTRS / guided_discovery"]            = sandbox_result["sandbox_ctrs_guided_discovery"]
        results["Sandbox CTRS / interpersonal_effectiveness"] = sandbox_result["sandbox_ctrs_interpersonal_effectiveness"]
        results["Sandbox CTRS Avg"]                           = sandbox_result["sandbox_ctrs_avg"]

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("\u2554" + "\u2550" * 58 + "\u2557")
    logger.info("\u2551  EVALUATION SUMMARY  [model: %-32s]\u2551", model.name)
    logger.info("\u2560" + "\u2550" * 58 + "\u2563")
    for metric, value in results.items():
        if not isinstance(value, float):
            continue
        if "F1" in metric:
            logger.info("\u2551  %-40s %10.4f  \u2551", metric, value)
        elif "CTRS" in metric and "Avg" in metric:
            logger.info("\u2551  %-40s %7.2f/6   \u2551", metric, value)
        elif "CTRS" in metric:
            logger.info("\u2551  %-40s %7.2f/6   \u2551", metric, value)
        elif "IG-PQA Mean" in metric:
            logger.info("\u2551  %-40s %7.4f b  \u2551", metric, value)
        elif "Ratio" in metric or "PQA" in metric:
            logger.info("\u2551  %-40s %9.1f%%  \u2551", metric, value * 100)
        else:
            logger.info("\u2551  %-40s %10.4f  \u2551", metric, value)
    logger.info("\u255a" + "\u2550" * 58 + "\u255d")
