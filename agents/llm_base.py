"""
llm_base.py
大模型调用接口基类。
支持任何符合 OpenAI Chat Completions 格式的 API（DeepSeek / GLM / GPT-4o 等）。

角色前缀规则（对应 .env 中的变量组）：
  THERAPIST_*   前端对话治疗师（System 1）
  SUPERVISOR_*  后台临床诊断器（System 2 / Diagnostician）
  JUDGE_*       评测裁判模型
  BASELINE_*    对比基线（单一通用大模型）
  LLM_*         通用回退默认值

所有节点通过 LLMClient.from_role(role) 工厂方法获取对应配置的实例。
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("cbt.llm_base")

# 已知角色前缀 -> .env 变量前缀映射
_ROLE_PREFIX: dict[str, str] = {
    "therapist":   "THERAPIST",
    "supervisor":  "SUPERVISOR",
    "diagnostician": "SUPERVISOR",  # Diagnostician 复用 SUPERVISOR 配置
    "judge":       "JUDGE",
    "baseline":    "BASELINE",
}


class LLMClient:
    """
    通用大模型调用客户端（OpenAI-compatible API）。

    推荐通过工厂方法创建：
        LLMClient.from_role("therapist")     # 读取 THERAPIST_* env 变量
        LLMClient.from_role("diagnostician") # 读取 SUPERVISOR_* env 变量
        LLMClient.from_role("judge")         # 读取 JUDGE_* env 变量
        LLMClient.from_role("baseline")      # 读取 BASELINE_* env 变量

    也可直接传参覆盖：
        LLMClient(api_key="...", base_url="...", model="...")
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        role_label: str = "default",
    ):
        self.role_label = role_label
        self.api_key = api_key or os.getenv("LLM_API_KEY") or ""
        self.base_url = (
            base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.temperature = float(
            temperature if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )
        self.timeout = int(
            timeout if timeout is not None
            else int(os.getenv("LLM_TIMEOUT", "120"))
        )
        self._session = requests.Session()
        logger.debug(
            "[LLMClient:%s] model=%s  base_url=%s  temperature=%.2f",
            self.role_label, self.model, self.base_url, self.temperature,
        )

    # ── 工厂方法：按角色从 .env 读取对应配置 ─────────────────────────────────

    @classmethod
    def from_role(
        cls,
        role: str,
        temperature: float | None = None,
        timeout: int | None = None,
    ) -> "LLMClient":
        """
        按角色名创建 LLMClient，自动从 .env 读取对应前缀的 API 配置。

        Parameters
        ----------
        role : "therapist" | "diagnostician" | "supervisor" | "judge" | "baseline"
               其他字符串则回退到 LLM_* 通用变量
        """
        prefix = _ROLE_PREFIX.get(role.lower(), "LLM")
        api_key  = os.getenv(f"{prefix}_API_KEY")  or os.getenv("LLM_API_KEY")  or ""
        base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
        model    = os.getenv(f"{prefix}_MODEL")    or os.getenv("LLM_MODEL")    or "gpt-4o-mini"
        logger.info(
            "[LLMClient] role=%-14s prefix=%s  model=%s",
            role, prefix, model,
        )
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout=timeout,
            role_label=role,
        )

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        发送多轮消息，返回 assistant 回复文本。

        Parameters
        ----------
        messages    : [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature : 若指定则覆盖实例默认值
        max_tokens  : 最大生成 token 数，None 时不传递
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # 记录完整请求消息（DEBUG 级别）
        logger.debug(
            "[LLMClient:%s] REQUEST messages (count=%d):\n%s",
            self.role_label,
            len(messages),
            "\n".join(
                f"  [{m['role'].upper()}]\n{m['content']}"
                for m in messages
            ),
        )

        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        logger.debug(
            "[LLMClient:%s] full reply:\n%s",
            self.role_label, content,
        )
        return content

    def simple_chat(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> str:
        """便捷方法：单次 system + user 对话。"""
        return self.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
        )

    # ── JSON 解析辅助 ─────────────────────────────────────────────────────────

    @staticmethod
    def extract_json(text: str) -> dict | None:
        """从模型回复中抽取第一个完整的 JSON 对象。"""
        # 整体尝试
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        # 去除 markdown 代码块
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # 提取最外层 {...}
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("[LLMClient] extract_json failed on: %s", text[:200])
        return None
