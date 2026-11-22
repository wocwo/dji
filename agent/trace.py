"""可观测性：把一次运行的事件写成 JSONL trace，便于回放、排障与评测。

trace 目录结构::

    .agent/traces/2026-11-18/tr_20261118_153000_ab12cd.jsonl
    .agent/traces/runs.jsonl        # 所有运行的索引（一行一条摘要）

写入失败不会中断 Agent，只是关闭追踪。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


class Tracer:
    """轻量 JSONL 追踪器。"""

    def __init__(self, directory: str = ".agent/traces", *, enabled: bool = True, include_llm_output: bool = True):
        self.directory = str(directory)
        self.enabled = bool(enabled and self.directory)
        self.include_llm_output = include_llm_output
        self.trace_id: str = ""
        self.path: Optional[Path] = None
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._warned = False

    # -- 生命周期 ---------------------------------------------------------
    def start_run(self, task: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.trace_id = f"tr_{stamp}_{uuid.uuid4().hex[:6]}"
        self.events = []
        if not self.enabled:
            return self.trace_id
        try:
            day = datetime.now().strftime("%Y-%m-%d")
            directory = Path(self.directory).expanduser() / day
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{self.trace_id}.jsonl"
        except Exception as exc:
            self._disable(f"trace 目录创建失败: {exc}")
            return self.trace_id
        self.log("run_start", payload={"task": task, "metadata": metadata or {}})
        return self.trace_id

    def log(self, event_type: str, *, step: int = 0, payload: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return
        event = {
            "trace_id": self.trace_id,
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "type": event_type,
            "step": step,
            "payload": payload or {},
        }
        with self._lock:
            self.events.append(event)
            if self.path is None:
                return
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            except Exception as exc:
                self._disable(f"trace 写入失败: {exc}")

    def finish(self, result: Any) -> None:
        if not self.enabled:
            return
        summary = {
            "status": getattr(result, "status", ""),
            "answer_chars": len(getattr(result, "answer", "") or ""),
            "steps": len(getattr(result, "steps", []) or []),
            "tool_calls": getattr(getattr(result, "usage", None), "tool_calls", 0),
            "duration_ms": round(float(getattr(result, "duration_ms", 0.0)), 3),
            "citations": len(getattr(result, "citations", []) or []),
            "forced_final": getattr(result, "forced_final", False),
            "warnings": list(getattr(result, "warnings", []) or []),
        }
        self.log("run_end", payload=summary)
        if self.path is None:
            return
        try:
            index = Path(self.directory).expanduser() / "runs.jsonl"
            index.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "trace_id": self.trace_id,
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "task": getattr(result, "task", ""),
                "status": summary["status"],
                "duration_ms": summary["duration_ms"],
                "steps": summary["steps"],
                "tool_calls": summary["tool_calls"],
                "citations": summary["citations"],
                "path": str(self.path),
            }
            with self._lock:
                with index.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.debug("trace 索引写入失败: %s", exc)

    def _disable(self, reason: str) -> None:
        self.enabled = False
        if not self._warned:
            logger.warning("追踪已关闭: %s", reason)
            self._warned = True


# ---------------------------------------------------------------------------
# 读取与回放
# ---------------------------------------------------------------------------

def read_trace(path: str) -> List[Dict[str, Any]]:
    """读取一个 trace 文件。"""
    events: List[Dict[str, Any]] = []
    file_path = Path(path).expanduser()
    if not file_path.exists():
        return events
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def list_traces(directory: str = ".agent/traces", *, limit: int = 20) -> List[Dict[str, Any]]:
    """读取运行索引，返回最近若干条。"""
    index = Path(directory).expanduser() / "runs.jsonl"
    if not index.exists():
        return []
    records: List[Dict[str, Any]] = []
    with index.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-max(1, limit):][::-1]


def render_trace(events: Iterable[Dict[str, Any]]) -> str:
    """把 trace 渲染成易读文本。"""
    lines: List[str] = []
    for event in events:
        etype = event.get("type", "?")
        step = event.get("step", 0)
        payload = event.get("payload") or {}
        if etype == "run_start":
            lines.append(f"[run] {payload.get('task', '')}")
        elif etype == "llm_response":
            text = str(payload.get("text", ""))
            lines.append(f"  [{step}] LLM: {text[:220]}{'…' if len(text) > 220 else ''}")
        elif etype == "tool_call":
            lines.append(f"  [{step}] CALL {payload.get('name')} {json.dumps(payload.get('arguments', {}), ensure_ascii=False)}")
        elif etype == "tool_result":
            lines.append(
                f"  [{step}] OBS  ok={payload.get('ok')} {str(payload.get('content', ''))[:200]}"
            )
        elif etype == "final":
            lines.append(f"  [{step}] FINAL {str(payload.get('answer', ''))[:220]}")
        elif etype in ("warning", "error", "budget"):
            lines.append(f"  [{step}] {etype.upper()}: {payload}")
        elif etype == "run_end":
            lines.append(f"[end] status={payload.get('status')} steps={payload.get('steps')} {payload.get('duration_ms')}ms")
    return "\n".join(lines)


__all__ = ["Tracer", "read_trace", "list_traces", "render_trace"]
