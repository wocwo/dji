"""稳定边界（Ports）：为后续引入 LangGraph / LangChain / LlamaIndex 预留接口。

当前运行时仍由 AgentRuntime 负责；这些 Protocol 只定义“能力契约”，
不改变现有行为。迁移时新增 adapter，而不是重写检索和证据链。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMClientPort(Protocol):
    name: str

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> Any: ...

    def health(self) -> Dict[str, Any]: ...


@runtime_checkable
class RetrieverPort(Protocol):
    name: str

    def search(self, query: str, top_k: int = 4, model: Optional[str] = None) -> List[Any]: ...

    def health(self) -> Dict[str, Any]: ...


@runtime_checkable
class RerankerPort(Protocol):
    def rank(self, query: str, candidates: Sequence[Any], top_k: int = 8) -> List[Any]: ...


@runtime_checkable
class VerifierPort(Protocol):
    def check(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class MemoryPort(Protocol):
    available: bool

    def append_turn(self, session_id: str, role: str, content: str) -> None: ...

    def get_turns(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]: ...


__all__ = [
    "LLMClientPort",
    "RetrieverPort",
    "RerankerPort",
    "VerifierPort",
    "MemoryPort",
]
