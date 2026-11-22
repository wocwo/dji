"""运行策略：预算、工具权限与人工确认。

预算与权限是 Agent 与「失控的循环」之间的护栏：
- 步数 / LLM 调用 / 工具调用 / 墙钟时间四重上限；
- 工具白名单、风险等级与确认回调；
- 所有限制都可通过配置调整，但默认值必须保守。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig
from .types import ToolCall


@dataclass
class Budget:
    """一次运行的资源预算。"""

    max_steps: int = 8
    max_tool_calls: int = 12
    max_llm_calls: int = 10
    max_seconds: float = 180.0
    started_at: float = field(default_factory=time.monotonic)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    def reason(self, *, steps: int, tool_calls: int, llm_calls: int) -> Optional[str]:
        """返回触发的停止原因；未触发返回 None。"""
        if steps >= self.max_steps:
            return "max_steps"
        if tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if llm_calls >= self.max_llm_calls:
            return "max_llm_calls"
        if self.elapsed() >= self.max_seconds:
            return "timeout"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_calls": self.max_llm_calls,
            "max_seconds": self.max_seconds,
            "elapsed_s": round(self.elapsed(), 3),
        }


@dataclass
class AgentPolicy:
    """工具权限与确认策略。"""

    allowed_tools: Optional[List[str]] = None
    allow_risky: bool = False
    confirm_moderate: bool = False
    tool_timeout: float = 30.0
    confirm: Optional[Callable[[ToolCall], bool]] = None
    max_steps: int = 8
    max_tool_calls: int = 12
    max_llm_calls: int = 10
    max_seconds: float = 180.0
    max_observation_chars: int = 4000
    max_consecutive_errors: int = 3

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        *,
        confirm: Optional[Callable[[ToolCall], bool]] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> "AgentPolicy":
        return cls(
            allowed_tools=allowed_tools if allowed_tools is not None else list(config.tools.enabled),
            allow_risky=config.tools.allow_risky,
            tool_timeout=config.tools.timeout,
            confirm=confirm,
            max_steps=config.agent.max_steps,
            max_tool_calls=config.agent.max_tool_calls,
            max_llm_calls=config.agent.max_llm_calls,
            max_seconds=config.agent.max_seconds,
            max_observation_chars=config.agent.max_observation_chars,
            max_consecutive_errors=config.agent.max_consecutive_errors,
        )

    def budget(self) -> Budget:
        return Budget(
            max_steps=self.max_steps,
            max_tool_calls=self.max_tool_calls,
            max_llm_calls=self.max_llm_calls,
            max_seconds=self.max_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools) if self.allowed_tools is not None else None,
            "allow_risky": self.allow_risky,
            "confirm_moderate": self.confirm_moderate,
            "tool_timeout": self.tool_timeout,
            "has_confirm": self.confirm is not None,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "max_llm_calls": self.max_llm_calls,
            "max_seconds": self.max_seconds,
            "max_observation_chars": self.max_observation_chars,
            "max_consecutive_errors": self.max_consecutive_errors,
        }


def interactive_confirm(call: ToolCall, *, input_fn: Callable[[str], str] = input) -> bool:
    """CLI 用的人工确认：默认拒绝，避免非交互环境下误执行。"""
    print(f"\n[确认] Agent 请求执行工具: {call.name}")
    print(f"       参数: {call.arguments}")
    try:
        answer = input_fn("是否允许？[y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes", "是")


__all__ = ["Budget", "AgentPolicy", "interactive_confirm"]
