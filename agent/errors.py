"""Agent 运行时异常定义。

设计原则：
- 所有异常都从 :class:`AgentError` 派生，便于上层统一捕获。
- 工具异常与 LLM 异常分开，运行时可以把它们转成 observation 继续推理，
  而不是直接中断整个任务。
"""

from __future__ import annotations


class AgentError(Exception):
    """Agent 基础异常。"""


class ConfigError(AgentError):
    """配置错误。"""


class LLMError(AgentError):
    """LLM 调用失败（网络、鉴权、响应格式等）。"""


class ToolError(AgentError):
    """工具执行失败。"""


class ToolNotFoundError(ToolError):
    """请求的工具不存在。"""


class ToolValidationError(ToolError):
    """工具参数校验失败。"""


class ToolTimeoutError(ToolError):
    """工具执行超时。"""


class PolicyError(AgentError):
    """被策略/权限拦截。"""


class KnowledgeBaseError(AgentError):
    """知识库不可用或检索失败。"""


class BudgetExceeded(AgentError):
    """预算（步数/工具调用/时间）耗尽。"""


class ParseError(AgentError):
    """模型输出无法解析为有效的动作。"""
