"""
webapp/core/persistence.py

会话本地持久化层 —— 将每轮对话后的 DialogueState 原子写入磁盘 JSON。

设计原则
--------
- 与 Flask、agents 完全解耦：仅依赖标准库，处理纯 dict 记录。
- 每个 session_id 对应一个 JSON 文件 <dir>/<session_id>.json。
- 原子写入：先写同目录临时文件，再 os.replace 覆盖，杜绝半截文件。
- 失败不抛出：持久化是旁路能力，任何 I/O 异常仅记录日志，绝不影响对话主流程。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("cbt.webapp.persistence")


class SessionStore:
    """会话记录的磁盘读写器（每会话一个 JSON 文件）。"""

    def __init__(self, directory: str | Path, enabled: bool = True):
        self.enabled = enabled
        self.dir = Path(directory)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    # ── 写入 ────────────────────────────────────────────────────────────────
    def save(self, record: dict[str, Any]) -> None:
        """原子写入一条会话记录。record 必须含 'session_id' 键。"""
        if not self.enabled:
            return
        sid = record.get("session_id")
        if not sid:
            logger.warning("[SessionStore] save skipped: record missing session_id")
            return

        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.dir, prefix=f".{sid}.", suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._path(sid))  # 原子覆盖
            tmp_path = None
        except Exception:
            logger.exception("[SessionStore] save failed for %s", str(sid)[:8])
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ── 读取 ────────────────────────────────────────────────────────────────
    def load(self, session_id: str) -> dict[str, Any] | None:
        """读取一条会话记录；不存在或损坏时返回 None。"""
        if not self.enabled:
            return None
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.exception("[SessionStore] load failed for %s", session_id[:8])
            return None

    # ── 列举 ────────────────────────────────────────────────────────────────
    def list_records(self) -> list[dict[str, Any]]:
        """
        读取目录下全部会话记录（跳过损坏/非法文件）。
        忽略以 '.' 开头的临时/隐藏文件。规模为本地单机研究量级，全量读取即可。
        """
        if not self.enabled or not self.dir.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in self.dir.glob("*.json"):
            if path.name.startswith("."):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                if isinstance(record, dict) and record.get("session_id"):
                    records.append(record)
            except Exception:
                logger.warning("[SessionStore] skip unreadable record: %s", path.name)
        return records
