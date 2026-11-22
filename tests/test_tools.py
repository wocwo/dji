"""工具系统测试：安全计算、参数校验、权限、超时、计划与笔记。"""

from __future__ import annotations

import time
from typing import Any

import pytest

from agent.errors import ToolValidationError
from agent.memory import AgentMemory
from agent.policy import AgentPolicy
from agent.tools.base import Tool, ToolContext, ToolRegistry, ToolSpec, validate_arguments
from agent.tools.builtin import (
    CalculatorTool,
    ClockTool,
    RecallNotesTool,
    SaveNoteTool,
    UpdatePlanTool,
    calculate,
)
from agent.types import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# 计算器
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1+2*3", 7),
        ("(5000-249)/1000", 4.751),
        ("2^10", 1024),
        ("sqrt(16)+2", 6),
        ("max(3, 9, 4)", 9),
        ("10 % 3", 1),
    ],
)
def test_calculator(expression, expected):
    result = CalculatorTool().run(ToolContext(), expression=expression)
    assert result.ok is True
    assert str(expected) in result.content
    assert result.data["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd').read()",
        "(1).__class__",
        "[1, 2, 3]",
        "lambda: 1",
        "1/0",
        "2**100000",
        "unknown_func(1)",
    ],
)
def test_calculator_rejects_unsafe(expression):
    result = CalculatorTool().run(ToolContext(), expression=expression)
    assert result.ok is False
    assert result.error


def test_calculate_rejects_empty():
    with pytest.raises(ValueError):
        calculate("   ")


# ---------------------------------------------------------------------------
# 测试用工具
# ---------------------------------------------------------------------------

class EchoTool(Tool):
    spec = ToolSpec(
        name="echo",
        description="回显",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "n": {"type": "integer", "default": 1},
            },
            "required": ["text"],
        },
    )

    def run(self, ctx: ToolContext, text: str, n: int = 1) -> Any:
        return {"content": text * n, "data": {"n": n}}


class RiskyTool(Tool):
    spec = ToolSpec(name="risky", description="危险操作", risk="risky")

    def run(self, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, content="done")


class SlowTool(Tool):
    spec = ToolSpec(name="slow", description="慢工具", risk="safe")

    def run(self, ctx: ToolContext) -> ToolResult:
        time.sleep(0.3)
        return ToolResult(ok=True, content="late")


class BoomTool(Tool):
    spec = ToolSpec(name="boom", description="会抛异常", risk="safe")

    def run(self, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("炸了")


# ---------------------------------------------------------------------------
# 参数校验与注册表
# ---------------------------------------------------------------------------

def test_validate_arguments_coercion():
    spec = EchoTool().spec
    args = validate_arguments(spec, {"text": "a", "n": "3"})
    assert args == {"text": "a", "n": 3}


def test_validate_arguments_missing_required():
    with pytest.raises(ToolValidationError):
        validate_arguments(EchoTool().spec, {})


def test_validate_arguments_wrong_type():
    with pytest.raises(ToolValidationError):
        validate_arguments(EchoTool().spec, {"text": 123, "n": 1})


def test_registry_normalizes_dict_output():
    registry = ToolRegistry([EchoTool()])
    try:
        result = registry.execute(ToolCall("echo", {"text": "ab", "n": 2}), ToolContext())
        assert result.ok is True
        assert result.content == "abab"
        assert result.data == {"n": 2}
        assert result.tool_name == "echo"
        assert result.latency_ms >= 0
    finally:
        registry.close()


def test_registry_unknown_tool():
    registry = ToolRegistry([EchoTool()])
    try:
        result = registry.execute(ToolCall("nope", {}), ToolContext())
        assert result.ok is False
        assert "工具不存在" in result.error
    finally:
        registry.close()


def test_registry_missing_argument():
    registry = ToolRegistry([EchoTool()])
    try:
        result = registry.execute(ToolCall("echo", {}), ToolContext())
        assert result.ok is False
        assert "缺少必填参数" in result.content
    finally:
        registry.close()


def test_registry_policy_allowlist():
    registry = ToolRegistry([EchoTool()])
    policy = AgentPolicy(allowed_tools=[])
    try:
        result = registry.execute(ToolCall("echo", {"text": "x"}), ToolContext(), policy)
        assert result.ok is False
        assert "不在允许列表" in result.content
    finally:
        registry.close()


def test_registry_risky_denied():
    registry = ToolRegistry([RiskyTool()])
    try:
        denied = registry.execute(ToolCall("risky", {}), ToolContext(), AgentPolicy())
        assert denied.ok is False
        allowed = registry.execute(
            ToolCall("risky", {}),
            ToolContext(),
            AgentPolicy(allow_risky=True),
        )
        assert allowed.ok is True
    finally:
        registry.close()


def test_registry_timeout():
    registry = ToolRegistry([SlowTool()], timeout=0.05)
    try:
        result = registry.execute(ToolCall("slow", {}), ToolContext())
        assert result.ok is False
        assert "超时" in result.content
    finally:
        registry.close()


def test_registry_wraps_exception():
    registry = ToolRegistry([BoomTool()])
    try:
        result = registry.execute(ToolCall("boom", {}), ToolContext())
        assert result.ok is False
        assert "炸了" in result.error
    finally:
        registry.close()


def test_registry_rejects_duplicate_registration():
    registry = ToolRegistry([EchoTool()])
    try:
        with pytest.raises(ValueError):
            registry.register(EchoTool())
    finally:
        registry.close()


# ---------------------------------------------------------------------------
# 计划 / 时间 / 笔记
# ---------------------------------------------------------------------------

def test_update_plan_preserves_status():
    ctx = ToolContext()
    tool = UpdatePlanTool()
    first = tool.run(ctx, items=["检索电池手册", "核对参数"])
    assert first.ok is True
    assert ctx.plan[0]["status"] == "pending"

    second = tool.run(
        ctx,
        items=[{"step": "检索电池手册", "status": "done"}, "核对参数"],
    )
    assert second.ok is True
    assert ctx.plan[0]["status"] == "done"
    assert "[x] 1." in second.content
    assert "[ ] 2." in second.content


def test_update_plan_rejects_empty():
    result = UpdatePlanTool().run(ToolContext(), items=[])
    assert result.ok is False


def test_clock_default_and_invalid():
    ok = ClockTool().run(ToolContext())
    assert ok.ok is True
    assert "Asia/Shanghai" in ok.content
    bad = ClockTool().run(ToolContext(), timezone="Mars/Olympus")
    assert bad.ok is False


def test_notes_roundtrip(tmp_path):
    memory = AgentMemory(str(tmp_path / "memory.db"))
    assert memory.available
    ctx = ToolContext(memory=memory)
    try:
        saved = SaveNoteTool().run(ctx, text="用户常用 Mavic 4 Pro", tags=["机型", "偏好"])
        assert saved.ok is True
        assert saved.data["note_id"] >= 1

        recalled = RecallNotesTool().run(ctx, query="Mavic", limit=5)
        assert recalled.ok is True
        assert "Mavic 4 Pro" in recalled.content

        by_tag = RecallNotesTool().run(ctx, tags=["偏好"])
        assert by_tag.ok is True
        assert by_tag.data["notes"]
    finally:
        memory.close()


def test_notes_without_memory():
    result = SaveNoteTool().run(ToolContext(), text="x")
    assert result.ok is False
    assert "未启用" in result.content
