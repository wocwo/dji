"""工具系统基础：ToolSpec / Tool / ToolContext / ToolRegistry。

设计要点：
- **声明式参数**：每个工具用 JSON Schema 子集声明参数，既用于提示词，也用于运行时校验。
- **统一结果**：工具可以返回字符串、字典或 ToolResult，注册表统一归一化。
- **可治理**：权限、风险等级、超时、异常都收敛在注册表，运行时只关心 observation。
- **可引用**：工具通过 ToolContext.cite() 分配全局唯一的 [S1] 引用编号。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..errors import (
    PolicyError,
    ToolNotFoundError,
    ToolTimeoutError,
    ToolValidationError,
)
from ..types import Citation, ToolCall, ToolResult


# ---------------------------------------------------------------------------
# 声明
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    """工具声明。"""

    name: str
    description: str
    parameters: Dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    risk: str = "safe"  # safe | moderate | risky
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "risk": self.risk,
            "tags": list(self.tags),
        }

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为 OpenAI function calling 的 tool schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool(ABC):
    """工具基类。子类实现 ``run(ctx, **kwargs)``。"""

    spec: ToolSpec

    @abstractmethod
    def run(self, ctx: "ToolContext", **kwargs: Any) -> Any:
        """执行工具。返回 str / dict / ToolResult 均可。"""

    # 便捷属性
    @property
    def name(self) -> str:
        return self.spec.name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Tool {self.spec.name}>"


# ---------------------------------------------------------------------------
# 运行上下文
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """一次 Agent 运行内的工具共享上下文。"""

    run_id: str = ""
    session_id: Optional[str] = None
    memory: Any = None
    kb: Any = None
    policy: Any = None
    plan: List[Dict[str, str]] = field(default_factory=list)
    requested_model: str = ""
    citations: Dict[str, Citation] = field(default_factory=dict)
    _next_ref: int = 0

    def cite(
        self,
        chunk_id: str,
        *,
        doc_id: str = "",
        title: str = "",
        section: str = "",
        page: Any = None,
        model: str = "",
        version: str = "",
        doc_type: str = "",
        source: str = "",
    ) -> str:
        """为一篇文档分配（或复用）引用编号，返回 ``S1``。"""
        key = str(chunk_id or f"{doc_id}:{page}:{title}:{section}")
        if key in self.citations:
            return self.citations[key].ref
        self._next_ref += 1
        ref = f"S{self._next_ref}"
        self.citations[key] = Citation(
            ref=ref,
            chunk_id=str(chunk_id or ""),
            doc_id=str(doc_id or ""),
            title=str(title or ""),
            section=str(section or ""),
            page=page,
            model=str(model or ""),
            version=str(version or ""),
            doc_type=str(doc_type or ""),
            source=str(source or ""),
        )
        return ref

    def all_citations(self) -> List[Citation]:
        def order(c: Citation) -> int:
            try:
                return int(c.ref.lstrip("S"))
            except ValueError:
                return 10 ** 6

        return sorted(self.citations.values(), key=order)


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

_TYPE_CHECKS: Dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def _coerce(value: Any, expected: str) -> Any:
    """对 LLM 常见的类型漂移做温和纠正。"""
    if expected == "integer" and isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (ValueError, TypeError):
            return value
    if expected == "number" and isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, TypeError):
            return value
    if expected == "boolean" and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "y", "是"):
            return True
        if lowered in ("false", "0", "no", "n", "否"):
            return False
        return value
    if expected == "array" and isinstance(value, str):
        # 支持 "a,b,c" 或 JSON 数组字符串
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [x.strip() for x in value.split(",") if x.strip()]
    return value


def validate_arguments(spec: ToolSpec, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """按 JSON Schema 子集校验并归一化参数。"""
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolValidationError(f"参数必须是 JSON 对象，收到 {type(arguments).__name__}")

    schema = spec.parameters or {}
    properties: Dict[str, Any] = schema.get("properties", {}) or {}
    required: Sequence[str] = schema.get("required", []) or []

    missing = [name for name in required if name not in arguments or arguments[name] is None]
    if missing:
        raise ToolValidationError(f"缺少必填参数: {', '.join(missing)}")

    cleaned: Dict[str, Any] = {}
    for name, meta in properties.items():
        if name not in arguments:
            if isinstance(meta, dict) and "default" in meta:
                cleaned[name] = meta["default"]
            continue
        value = arguments[name]
        expected = meta.get("type") if isinstance(meta, dict) else None
        if expected:
            value = _coerce(value, expected)
            checker = _TYPE_CHECKS.get(expected)
            if checker and not checker(value):
                raise ToolValidationError(
                    f"参数 {name} 类型应为 {expected}，收到 {type(value).__name__}"
                )
        enum = meta.get("enum") if isinstance(meta, dict) else None
        if enum and value not in enum:
            raise ToolValidationError(f"参数 {name} 必须是 {enum} 之一")
        cleaned[name] = value

    # 未声明的参数直接忽略，避免模型偶发多余字段导致整次调用失败
    return cleaned


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------

class ToolRegistry:
    """工具注册表 + 安全执行器。"""

    def __init__(self, tools: Optional[Sequence[Tool]] = None, *, timeout: float = 30.0, max_workers: int = 4):
        self._tools: Dict[str, Tool] = {}
        self.default_timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent-tool")
        for tool in tools or []:
            self.register(tool)

    # -- 注册 -------------------------------------------------------------
    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError(f"register 需要 Tool 实例，收到 {type(tool).__name__}")
        name = tool.spec.name
        if not name:
            raise ValueError("工具必须有 name")
        if name in self._tools:
            raise ValueError(f"工具重复注册: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"工具不存在: {name}（可用: {', '.join(self.names())}）")
        return tool

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def specs(self, enabled: Optional[Sequence[str]] = None) -> List[ToolSpec]:
        if enabled is None:
            return [t.spec for t in self._tools.values()]
        return [self._tools[n].spec for n in enabled if n in self._tools]

    def schema(self, enabled: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        return [s.to_openai_schema() for s in self.specs(enabled)]

    def describe(self, enabled: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.specs(enabled)]

    # -- 执行 -------------------------------------------------------------
    def execute(
        self,
        call: ToolCall,
        ctx: ToolContext,
        policy: Any = None,
        *,
        enabled: Optional[Sequence[str]] = None,
    ) -> ToolResult:
        """执行一次工具调用。任何异常都会被转换成 ``ok=False`` 的结果。"""
        start = time.time()

        def finish(result: ToolResult) -> ToolResult:
            result.tool_name = call.name
            result.latency_ms = (time.time() - start) * 1000
            return result

        try:
            tool = self.get(call.name)
            if enabled is not None and call.name not in enabled:
                raise PolicyError(f"工具 {call.name} 未在启用列表中")

            self._check_policy(tool, call, policy)

            try:
                args = validate_arguments(tool.spec, call.arguments or {})
            except ToolValidationError as exc:
                return finish(ToolResult(ok=False, content=f"参数错误：{exc}", error=str(exc)))

            before = set(ctx.citations.keys())
            timeout = getattr(policy, "tool_timeout", None) or self.default_timeout
            future: Future = self._executor.submit(tool.run, ctx, **args)
            try:
                output = future.result(timeout=timeout)
            except FuturesTimeout:
                future.cancel()
                raise ToolTimeoutError(f"工具 {call.name} 执行超时（>{timeout:.0f}s）")

            result = _normalize_output(call.name, output)
            # 收集本步新增的引用
            new_keys = [k for k in ctx.citations.keys() if k not in before]
            if new_keys:
                existing = {c.chunk_id for c in result.citations}
                for key in new_keys:
                    citation = ctx.citations[key]
                    if citation.chunk_id not in existing:
                        result.citations.append(citation)
                        existing.add(citation.chunk_id)
            return finish(result)

        except PolicyError as exc:
            return finish(ToolResult(ok=False, content=f"调用被策略拦截：{exc}", error=str(exc)))
        except ToolNotFoundError as exc:
            return finish(ToolResult(ok=False, content=f"工具不可用：{exc}", error=str(exc)))
        except ToolTimeoutError as exc:
            return finish(ToolResult(ok=False, content=f"工具超时：{exc}", error=str(exc)))
        except Exception as exc:  # 工具内部异常统一转 observation
            return finish(ToolResult(ok=False, content=f"工具执行失败：{exc}", error=f"{type(exc).__name__}: {exc}"))

    def _check_policy(self, tool: Tool, call: ToolCall, policy: Any) -> None:
        if policy is None:
            return
        allowed = getattr(policy, "allowed_tools", None)
        if allowed is not None and call.name not in set(allowed):
            raise PolicyError(f"工具 {call.name} 不在允许列表")
        if tool.spec.risk == "risky":
            if not getattr(policy, "allow_risky", False):
                raise PolicyError(f"工具 {call.name} 属于高风险操作，当前策略未授权")
            confirm = getattr(policy, "confirm", None)
            if callable(confirm) and not confirm(call):
                raise PolicyError(f"用户未确认工具 {call.name} 的执行")
        elif tool.spec.risk == "moderate" and getattr(policy, "confirm_moderate", False):
            confirm = getattr(policy, "confirm", None)
            if callable(confirm) and not confirm(call):
                raise PolicyError(f"用户未确认工具 {call.name} 的执行")

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


def _normalize_output(name: str, output: Any) -> ToolResult:
    """把工具的任意返回值归一化为 ToolResult。"""
    if isinstance(output, ToolResult):
        return output
    if output is None:
        return ToolResult(ok=True, content="（工具执行完成，无输出）", tool_name=name)
    if isinstance(output, str):
        return ToolResult(ok=True, content=output, tool_name=name)
    if isinstance(output, dict):
        content = output.get("content") or output.get("text") or output.get("answer")
        data = output.get("data")
        if content is None:
            content = json.dumps(output, ensure_ascii=False)
        if data is None:
            data = {k: v for k, v in output.items() if k not in ("content", "text", "answer")}
        return ToolResult(ok=True, content=str(content), data=data if isinstance(data, dict) else {"value": data}, tool_name=name)
    if isinstance(output, (list, tuple)):
        return ToolResult(ok=True, content="\n".join(str(x) for x in output), tool_name=name)
    return ToolResult(ok=True, content=str(output), tool_name=name)
