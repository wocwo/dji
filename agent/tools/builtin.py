"""内置工具：计算器、时间、计划、笔记。

这些工具不依赖 RAG，用于补齐 Agent 在“非手册事实”上的能力，
并演示如何编写一个符合 ToolSpec 的可治理工具。
"""

from __future__ import annotations

import ast
import json
import math
import operator
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..types import ToolResult
from .base import Tool, ToolContext, ToolSpec


# ---------------------------------------------------------------------------
# 计算器
# ---------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"不支持的常量: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大（上限 1000）")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ValueError("只允许使用 sqrt/abs/round/min/max/pow/log/exp/floor/ceil 等函数")
        if node.keywords:
            raise ValueError("不支持关键字参数")
        args = [_safe_eval(arg) for arg in node.args]
        return _FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError(f"不支持的表达式元素: {type(node).__name__}")


def calculate(expression: str) -> Any:
    """安全计算四则运算表达式。"""
    normalized = (
        str(expression)
        .replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("^", "**")
        .strip()
    )
    if not normalized:
        raise ValueError("表达式为空")
    if len(normalized) > 200:
        raise ValueError("表达式过长（>200 字符）")
    tree = ast.parse(normalized, mode="eval")
    result = _safe_eval(tree)
    if isinstance(result, float):
        if not math.isfinite(result):
            raise ValueError("计算结果不是有限数值")
        if abs(result) > 1e15:
            raise ValueError("计算结果过大")
    return result


class CalculatorTool(Tool):
    """四则运算/常见数学函数。"""

    spec = ToolSpec(
        name="calculator",
        description=(
            "执行数学计算（四则运算、幂、开方、对数等）。"
            "当用户问题涉及载重、续航换算、价格、比例、单位换算等算术时使用。"
            "不要用它处理手册事实检索。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 (5000-249)/1000 或 sqrt(16)+2",
                }
            },
            "required": ["expression"],
        },
        risk="safe",
        tags=["math", "utility"],
    )

    def run(self, ctx: ToolContext, expression: str) -> ToolResult:
        try:
            result = calculate(expression)
        except Exception as exc:
            return ToolResult(ok=False, content=f"计算失败：{exc}", error=str(exc))
        display = int(result) if isinstance(result, float) and result.is_integer() else result
        return ToolResult(
            ok=True,
            content=f"计算结果：{expression} = {display}",
            data={"expression": expression, "result": display},
        )


# ---------------------------------------------------------------------------
# 时间
# ---------------------------------------------------------------------------

_WEEKDAYS = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


class ClockTool(Tool):
    """当前时间（带时区）。"""

    spec = ToolSpec(
        name="clock",
        description=(
            "获取当前日期与时间。当问题涉及“今天/现在/最近/保修期还剩多久”等时间相关计算时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "IANA 时区名，例如 Asia/Shanghai、UTC；默认 Asia/Shanghai",
                    "default": "Asia/Shanghai",
                }
            },
            "required": [],
        },
        risk="safe",
        tags=["utility"],
    )

    def run(self, ctx: ToolContext, timezone: str = "Asia/Shanghai") -> ToolResult:
        try:
            tz = ZoneInfo(timezone or "Asia/Shanghai")
        except ZoneInfoNotFoundError:
            return ToolResult(ok=False, content=f"未知时区：{timezone}", error="invalid timezone")
        now = datetime.now(tz)
        weekday = _WEEKDAYS[now.weekday()]
        content = f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{weekday}，{timezone}）"
        return ToolResult(
            ok=True,
            content=content,
            data={"iso": now.isoformat(), "timezone": timezone, "weekday": weekday},
        )


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------

class UpdatePlanTool(Tool):
    """维护一个显式的执行计划（todo list）。"""

    spec = ToolSpec(
        name="update_plan",
        description=(
            "创建或更新本轮任务的执行计划。适合问题需要多步检索/多机型对比时，"
            "先列出 2-6 个步骤，完成一步后更新状态。简单问题不需要调用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "计划项列表；每项是字符串，或 {step, status} 对象，status 为 pending/done",
                    "items": {"type": "string"},
                }
            },
            "required": ["items"],
        },
        risk="safe",
        tags=["planning"],
    )

    def run(self, ctx: ToolContext, items: Any) -> ToolResult:
        if not isinstance(items, list) or not items:
            return ToolResult(ok=False, content="计划不能为空", error="items must be a non-empty list")
        if len(items) > 12:
            return ToolResult(ok=False, content="计划项过多（最多 12 项）", error="too many items")

        existing = {item.get("step"): item.get("status", "pending") for item in ctx.plan}
        plan: List[Dict[str, str]] = []
        for raw in items:
            if isinstance(raw, str):
                step = raw.strip()
                status = existing.get(step, "pending")
            elif isinstance(raw, dict):
                step = str(raw.get("step") or raw.get("text") or raw.get("title") or "").strip()
                status = str(raw.get("status") or existing.get(step, "pending"))
            else:
                continue
            if not step:
                continue
            if status not in ("pending", "done", "blocked"):
                status = "pending"
            plan.append({"step": step, "status": status})

        if not plan:
            return ToolResult(ok=False, content="计划项均无效", error="empty plan")
        ctx.plan = plan

        done = sum(1 for item in plan if item["status"] == "done")
        lines = [f"执行计划（{done}/{len(plan)} 完成）："]
        for idx, item in enumerate(plan, start=1):
            mark = "x" if item["status"] == "done" else ("!" if item["status"] == "blocked" else " ")
            lines.append(f"- [{mark}] {idx}. {item['step']}")
        return ToolResult(ok=True, content="\n".join(lines), data={"plan": plan})


# ---------------------------------------------------------------------------
# 笔记（长期记忆）
# ---------------------------------------------------------------------------

class SaveNoteTool(Tool):
    """把用户偏好/结论写入长期记忆。"""

    spec = ToolSpec(
        name="save_note",
        description=(
            "把值得跨会话记住的信息写入长期记忆，例如用户常用机型、已确认的结论、待办事项。"
            "日常问答不要滥用；只保存真正有复用价值的信息。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "笔记内容"},
                "tags": {
                    "type": "array",
                    "description": "标签列表，例如 ['Mavic 4 Pro', '电池']",
                    "items": {"type": "string"},
                },
            },
            "required": ["text"],
        },
        risk="moderate",
        tags=["memory"],
    )

    def run(self, ctx: ToolContext, text: str, tags: Optional[List[str]] = None) -> ToolResult:
        if ctx.memory is None:
            return ToolResult(ok=False, content="长期记忆未启用，无法保存笔记", error="memory disabled")
        if hasattr(ctx.memory, "available") and not ctx.memory.available:
            return ToolResult(ok=False, content="长期记忆不可用（数据库初始化失败）", error="memory unavailable")
        text = str(text).strip()
        if not text:
            return ToolResult(ok=False, content="笔记内容为空", error="empty note")
        note_id = ctx.memory.add_note(text, tags or [])
        return ToolResult(
            ok=True,
            content=f"已保存笔记 #{note_id}：{text[:120]}",
            data={"note_id": note_id, "text": text, "tags": tags or []},
        )


class RecallNotesTool(Tool):
    """检索长期记忆中的笔记。"""

    spec = ToolSpec(
        name="recall_notes",
        description=(
            "检索长期记忆中的笔记。当用户提到“之前说过/上次/我的偏好/记住的内容”时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索关键词（可选）"},
                "tags": {
                    "type": "array",
                    "description": "按标签过滤（可选）",
                    "items": {"type": "string"},
                },
                "limit": {"type": "integer", "description": "返回条数，默认 5", "default": 5},
            },
            "required": [],
        },
        risk="safe",
        tags=["memory"],
    )

    def run(
        self,
        ctx: ToolContext,
        query: str = "",
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> ToolResult:
        if ctx.memory is None:
            return ToolResult(ok=False, content="长期记忆未启用", error="memory disabled")
        if hasattr(ctx.memory, "available") and not ctx.memory.available:
            return ToolResult(ok=False, content="长期记忆不可用（数据库初始化失败）", error="memory unavailable")
        limit = max(1, min(int(limit or 5), 20))
        notes = ctx.memory.search_notes(query=query, tags=tags, limit=limit)
        if not notes:
            return ToolResult(ok=True, content="长期记忆中没有匹配的笔记。", data={"notes": []})
        lines = [f"找到 {len(notes)} 条笔记："]
        for note in notes:
            tags_text = f"（标签：{', '.join(note.get('tags') or [])}）" if note.get("tags") else ""
            lines.append(f"- #{note['id']} [{note.get('created_at', '')}] {note['text']}{tags_text}")
        return ToolResult(ok=True, content="\n".join(lines), data={"notes": notes})


__all__ = [
    "CalculatorTool",
    "ClockTool",
    "UpdatePlanTool",
    "SaveNoteTool",
    "RecallNotesTool",
    "calculate",
]
