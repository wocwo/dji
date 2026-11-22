"""DJI 手册 Agent —— 在原 RAG 工程之上的多步工具调用 Agent。

公开 API::

    from agent import Agent, AgentConfig, load_config
    from agent.tools import Tool, ToolSpec, ToolRegistry

核心对象：
- :class:`Agent`：装配好的门面（LLM + 知识库 + 工具 + 记忆 + 追踪）。
- :class:`AgentRuntime`：ReAct 循环。
- :class:`AgentResult`：结构化运行结果（步骤、引用、用量、状态）。
"""

from __future__ import annotations

from .agent import Agent
from .claim_verifier import ClaimVerifier, ClaimReport, ClaimVerdict
from .config import AgentConfig, load_config
from .errors import AgentError, BudgetExceeded, ConfigError, LLMError, ToolError
from .grounding import GroundingReport, GroundingVerifier, normalize_citations
from .knowledge import KnowledgeBase, Passage, create_knowledge_base
from .llm import LLMClient, LLMResponse, create_llm
from .memory import AgentMemory
from .parser import ParsedAction, extract_citation_refs, parse_action
from .policy import AgentPolicy, Budget
from .runtime import AgentRuntime
from .tools import Tool, ToolContext, ToolRegistry, ToolSpec, build_tool_registry
from .trace import Tracer, list_traces, read_trace, render_trace
from .types import (
    AgentEvent,
    AgentResult,
    AgentStep,
    Citation,
    Message,
    ToolCall,
    ToolResult,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "load_config",
    "AgentRuntime",
    "AgentResult",
    "AgentEvent",
    "AgentStep",
    "AgentPolicy",
    "AgentMemory",
    "AgentError",
    "BudgetExceeded",
    "ConfigError",
    "LLMError",
    "ToolError",
    "Budget",
    "Citation",
    "Message",
    "ToolCall",
    "ToolResult",
    "Usage",
    "Tool",
    "ToolSpec",
    "ToolContext",
    "ToolRegistry",
    "build_tool_registry",
    "KnowledgeBase",
    "Passage",
    "create_knowledge_base",
    "LLMClient",
    "LLMResponse",
    "create_llm",
    "ParsedAction",
    "parse_action",
    "extract_citation_refs",
    "normalize_citations",
    "GroundingVerifier",
    "GroundingReport",
    "ClaimVerifier",
    "ClaimReport",
    "ClaimVerdict",
    "Tracer",
    "read_trace",
    "list_traces",
    "render_trace",
    "__version__",
]
