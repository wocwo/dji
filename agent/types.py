"""Agent 核心数据类型。

这些类型是运行时、工具、记忆、追踪与 CLI/HTTP 层之间的公共契约。
所有类型都提供 ``to_dict()``，便于序列化为 JSON / SSE 事件。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# 基础工具函数
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """返回带时区的 ISO 时间字符串。"""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def new_id(prefix: str = "") -> str:
    """生成短 ID。"""
    value = uuid.uuid4().hex[:12]
    return f"{prefix}{value}" if prefix else value


def _jsonable(value: Any) -> Any:
    """尽力把任意对象转换成可 JSON 序列化的结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except Exception:  # pragma: no cover - 防御性
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# 消息与调用
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """一次工具调用请求。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call_"))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        args = data.get("arguments", data.get("args", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"input": args}
        if not isinstance(args, dict):
            args = {"input": args}
        return cls(
            name=str(data.get("name", data.get("tool", ""))),
            arguments=args,
            id=str(data.get("id") or new_id("call_")),
        )


@dataclass
class Message:
    """对话消息，兼容文本协议与原生 tool calling 两种形态。"""

    role: str  # system / user / assistant / tool
    content: str = ""
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data

    def to_wire(self) -> Dict[str, Any]:
        """转换为 OpenAI / Ollama Chat API 的消息格式。"""
        data: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


# ---------------------------------------------------------------------------
# 用量与引用
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """一次运行内的资源用量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_calls += other.llm_calls
        self.tool_calls += other.tool_calls

    def merge(self, other: "Usage") -> None:
        """``add`` 的别名，便于运行时表达“合并一次调用的用量”。"""
        self.add(other)

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
        }


@dataclass
class Citation:
    """一条可追溯的知识来源。ref 在一次运行内全局唯一（S1、S2...）。"""

    ref: str
    chunk_id: str = ""
    doc_id: str = ""
    title: str = ""
    section: str = ""
    page: Any = None
    model: str = ""
    version: str = ""
    doc_type: str = ""
    source: str = ""

    def label(self) -> str:
        parts: List[str] = []
        if self.title:
            parts.append(self.title)
        elif self.doc_id:
            parts.append(self.doc_id)
        if self.page not in (None, "", 0):
            parts.append(f"第{self.page}页")
        if self.section:
            parts.append(str(self.section))
        return " · ".join(parts) if parts else (self.chunk_id or self.ref)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "page": self.page,
            "model": self.model,
            "version": self.version,
            "doc_type": self.doc_type,
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# 工具执行结果
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """工具执行结果。

    ``content`` 是给模型看的文本（observation）；``data`` 是给程序用的结构化结果。
    """

    ok: bool = True
    content: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0
    truncated: bool = False
    tool_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "content": self.content,
            "data": _jsonable(self.data),
            "citations": [c.to_dict() for c in self.citations],
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
            "truncated": self.truncated,
            "tool_name": self.tool_name,
        }


# ---------------------------------------------------------------------------
# 步骤、事件与最终结果
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """Agent 循环中的一步。"""

    index: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    result: Optional[ToolResult] = None
    raw_response: str = ""
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "result": self.result.to_dict() if self.result else None,
            "raw_response": self.raw_response,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass
class AgentEvent:
    """运行过程中的事件，用于 CLI 进度展示、SSE 流式输出与追踪。"""

    type: str
    step: int = 0
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "step": self.step,
            "payload": _jsonable(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass
class AgentResult:
    """一次 Agent 运行的完整结果。"""

    task: str
    answer: str = ""
    status: str = "ok"  # ok / max_steps / max_tool_calls / timeout / error / no_answer
    steps: List[AgentStep] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    trace_id: str = ""
    started_at: str = field(default_factory=now_iso)
    duration_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    forced_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Message] = field(default_factory=list)

    # -- 便捷访问 ---------------------------------------------------------
    @property
    def succeeded(self) -> bool:
        return self.status == "ok" and bool(self.answer.strip())

    @property
    def tool_calls(self) -> List[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    def to_dict(self, *, include_steps: bool = True, include_transcript: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "task": self.task,
            "answer": self.answer,
            "status": self.status,
            "succeeded": self.succeeded,
            "citations": [c.to_dict() for c in self.citations],
            "usage": self.usage.to_dict(),
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 3),
            "warnings": self.warnings,
            "forced_final": self.forced_final,
            "metadata": _jsonable(self.metadata),
        }
        if include_steps:
            data["steps"] = [s.to_dict() for s in self.steps]
        if include_transcript:
            data["transcript"] = [m.to_dict() for m in self.transcript]
        return data

    def to_markdown(self) -> str:
        """渲染为便于阅读的 Markdown（CLI / 报告用）。"""
        lines = [
            f"# Agent 运行结果",
            "",
            f"- **任务**：{self.task}",
            f"- **状态**：{self.status}",
            f"- **耗时**：{self.duration_ms / 1000:.2f}s",
            f"- **步数**：{len(self.steps)}（工具调用 {self.usage.tool_calls} 次）",
            f"- **Trace**：`{self.trace_id}`",
            "",
            "## 回答",
            "",
            self.answer or "（无输出）",
        ]
        if self.citations:
            lines += ["", "## 引用", ""]
            for c in self.citations:
                lines.append(f"- [{c.ref}] {c.label()}")
        if self.steps:
            lines += ["", "## 执行轨迹", ""]
            for s in self.steps:
                if s.tool_call:
                    args = json.dumps(s.tool_call.arguments, ensure_ascii=False)
                    lines.append(f"{s.index}. `{s.tool_call.name}`({args})")
                    if s.result and not s.result.ok:
                        lines.append(f"   - 失败：{s.result.error}")
                elif s.error:
                    lines.append(f"{s.index}. 解析失败：{s.error}")
        if self.warnings:
            lines += ["", "## 警告", ""] + [f"- {w}" for w in self.warnings]
        return "\n".join(lines)
