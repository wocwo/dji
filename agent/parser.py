"""模型动作解析。

支持三种常见形态：
1. 文本协议 JSON：``{"thought": ..., "action": ..., "action_input": ...}``
2. ReAct 文本：``Action: manual_search`` + ``Action Input: {...}``
3. 纯文本最终回答（可配置是否接受，作为对本地弱模型的兜底）

解析器必须“宽进严出”：模型经常加代码围栏、前后解释、单引号、尾逗号，
这里尽量恢复出结构，同时把无法恢复的情况明确标记为 invalid，交由运行时纠错。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .types import ToolCall


@dataclass
class ParsedAction:
    """解析后的动作。"""

    kind: str  # tool | final | invalid
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    final_answer: str = ""
    error: str = ""
    raw: str = ""
    soft: bool = False  # 纯文本被当作最终答案
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "thought": self.thought,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "final_answer": self.final_answer,
            "error": self.error,
            "soft": self.soft,
        }


# ---------------------------------------------------------------------------
# JSON 提取
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _iter_json_candidates(text: str) -> List[str]:
    """收集可能的 JSON 片段：围栏、整体、平衡括号扫描。"""
    candidates: List[str] = []

    for match in _FENCE_RE.finditer(text):
        block = match.group(1).strip()
        if block:
            candidates.append(block)

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    # 平衡括号扫描（跳过字符串内的括号）
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    if candidate not in candidates:
                        candidates.append(candidate)
                    break
    return candidates


def _loads_relaxed(candidate: str) -> Optional[Dict[str, Any]]:
    """尝试解析 JSON，允许尾逗号等常见小瑕疵。"""
    attempts = [candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)]
    for attempt in attempts:
        try:
            value = json.loads(attempt)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _as_arguments(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"input": value}
    if value is None:
        return {}
    return {"input": value}


def _coerce_tool_payload(data: Dict[str, Any]) -> Optional[ParsedAction]:
    """从 JSON 对象中识别工具调用。"""
    if not isinstance(data, dict):
        return None

    thought = str(data.get("thought") or data.get("reasoning") or "").strip()

    # 形态 A：{"action": "name", "action_input": {...}}
    action = data.get("action", data.get("tool", data.get("name")))
    if isinstance(action, dict):  # {"action": {"name": ..., "arguments": ...}}
        inner = action
        name = inner.get("name") or inner.get("tool")
        args = _as_arguments(inner.get("arguments", inner.get("input", {})))
        if name:
            return ParsedAction(
                kind="tool",
                thought=thought,
                tool_call=ToolCall(name=str(name), arguments=args),
                payload=data,
            )
        return None

    if isinstance(action, str) and action.strip():
        name = action.strip()
        args_raw = data.get("action_input", data.get("arguments", data.get("args", data.get("input", {}))))
        return ParsedAction(
            kind="tool",
            thought=thought,
            tool_call=ToolCall(name=name, arguments=_as_arguments(args_raw)),
            payload=data,
        )

    # 形态 B：{"tool": "name", "arguments": {...}} 已被上面覆盖；再兜底
    if isinstance(data.get("tool"), str) and data.get("tool"):
        return ParsedAction(
            kind="tool",
            thought=thought,
            tool_call=ToolCall(
                name=str(data["tool"]),
                arguments=_as_arguments(data.get("arguments", data.get("args", {}))),
            ),
            payload=data,
        )
    return None


def _coerce_final_payload(data: Dict[str, Any]) -> Optional[ParsedAction]:
    """从 JSON 对象中识别最终答案。"""
    for key in ("final_answer", "answer", "final", "response"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            thought = str(data.get("thought") or data.get("reasoning") or "").strip()
            return ParsedAction(
                kind="final",
                thought=thought,
                final_answer=value.strip(),
                payload=data,
            )
    return None


# ---------------------------------------------------------------------------
# ReAct 文本格式
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"Action\s*[:：]\s*([A-Za-z_][\w\-\.]*)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action\s*Input\s*[:：]\s*(\{.*?\})\s*(?:\n|$)", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)


def _parse_react_text(text: str) -> Optional[ParsedAction]:
    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        answer = final_match.group(1).strip()
        if answer:
            return ParsedAction(kind="final", final_answer=answer)

    action_match = _ACTION_RE.search(text)
    if not action_match:
        return None
    name = action_match.group(1)
    args: Dict[str, Any] = {}
    input_match = _ACTION_INPUT_RE.search(text)
    if input_match:
        args = _as_arguments(input_match.group(1))
    thought_match = re.search(r"Thought\s*[:：]\s*(.+)", text, re.IGNORECASE)
    thought = thought_match.group(1).strip() if thought_match else ""
    return ParsedAction(
        kind="tool",
        thought=thought,
        tool_call=ToolCall(name=name, arguments=args),
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def parse_action(text: str, *, plain_text_final: bool = True) -> ParsedAction:
    """把模型输出解析为工具调用 / 最终答案 / 无效动作。"""
    raw = text or ""
    if not raw.strip():
        return ParsedAction(kind="invalid", error="模型输出为空", raw=raw)

    errors: List[str] = []
    for candidate in _iter_json_candidates(raw):
        data = _loads_relaxed(candidate)
        if data is None:
            continue
        tool_action = _coerce_tool_payload(data)
        if tool_action is not None:
            tool_action.raw = raw
            return tool_action
        final_action = _coerce_final_payload(data)
        if final_action is not None:
            final_action.raw = raw
            return final_action
        errors.append("JSON 中缺少 action 或 final_answer 字段")

    react = _parse_react_text(raw)
    if react is not None:
        react.raw = raw
        return react

    if plain_text_final:
        return ParsedAction(
            kind="final",
            final_answer=raw.strip(),
            raw=raw,
            soft=True,
        )

    return ParsedAction(
        kind="invalid",
        error="；".join(errors) or "无法解析出 JSON 动作",
        raw=raw,
    )


_CITE_RE = re.compile(
    r"[\[［]\s*(S\d+(?:\s*[,，、]\s*S\d+)*)\s*[\]］]",
    re.IGNORECASE,
)


def extract_citation_refs(answer: str) -> List[str]:
    """从最终答案中提取 ``[S1]`` / ``[S1, S2]`` 引用编号。

    同时兼容全角括号 ``［S1］``、中文逗号和 ``[S1][S2]`` 连写。
    """
    refs: List[str] = []
    for match in _CITE_RE.finditer(answer or ""):
        for part in re.split(r"[,，、]", match.group(1)):
            ref = part.strip().upper()
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def count_json_objects(text: str) -> int:
    """调试辅助：统计文本中的 JSON 对象数量。"""
    return len(_iter_json_candidates(text))
