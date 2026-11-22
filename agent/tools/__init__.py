"""工具包：注册表、内置工具与 RAG 工具。"""

from __future__ import annotations

from typing import List, Optional

from ..config import AgentConfig
from ..knowledge import KnowledgeBase
from ..types import ToolResult
from .base import Tool, ToolContext, ToolRegistry, ToolSpec, validate_arguments
from .builtin import (
    CalculatorTool,
    ClockTool,
    RecallNotesTool,
    SaveNoteTool,
    UpdatePlanTool,
)
from .rag import ManualFetchTool, ManualSearchTool


def build_tool_registry(
    config: AgentConfig,
    kb: Optional[KnowledgeBase] = None,
    memory: Optional[object] = None,
) -> ToolRegistry:
    """根据配置构建工具注册表。

    - 没有知识库时不注册 ``manual_search`` / ``manual_fetch``；
    - 只注册 ``tools.enabled`` 中列出的工具；
    - 未识别的工具名会被忽略（配置向前兼容）。
    """
    candidates: List[Tool] = [
        CalculatorTool(),
        ClockTool(),
        UpdatePlanTool(),
        SaveNoteTool(),
        RecallNotesTool(),
    ]
    if kb is not None:
        candidates = [
            ManualSearchTool(
                kb,
                default_top_k=config.knowledge.top_k,
                max_passage_chars=config.knowledge.max_passage_chars,
            ),
            ManualFetchTool(kb),
        ] + candidates

    registry = ToolRegistry(timeout=config.tools.timeout)
    enabled = set(config.tools.enabled)
    for tool in candidates:
        if tool.spec.name in enabled:
            registry.register(tool)
    return registry


__all__ = [
    "Tool",
    "ToolSpec",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "validate_arguments",
    "ManualSearchTool",
    "ManualFetchTool",
    "CalculatorTool",
    "ClockTool",
    "UpdatePlanTool",
    "SaveNoteTool",
    "RecallNotesTool",
    "build_tool_registry",
]
