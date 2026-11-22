"""Agent 门面：装配 LLM、知识库、工具、记忆、追踪与运行时。

典型用法::

    from agent import Agent
    agent = Agent.from_config("configs/agent.yaml")
    result = agent.ask("Mavic 4 Pro 的智能飞行电池怎么保养？")
    print(result.answer)

    for event in agent.stream("固件升级失败怎么办？"):
        print(event.type, event.payload)
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from .config import AgentConfig, load_config
from .knowledge import create_knowledge_base
from .llm import LLMClient, create_llm
from .memory import AgentMemory
from .policy import AgentPolicy
from .runtime import AgentRuntime, EventCallback, TraceFactory
from .tools import ToolRegistry, build_tool_registry
from .types import AgentEvent, AgentResult, Message

logger = logging.getLogger(__name__)


class Agent:
    """装配好的 Agent 实例。线程安全性：每次 ``ask`` 使用独立 trace/上下文，
    共享的 LLM 客户端、SQLite 记忆与工具注册表均做了并发保护；
    但对同一会话的并发写入仍建议由调用方串行化。
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        *,
        llm: Optional[LLMClient] = None,
        kb: Any = None,
        memory: Optional[AgentMemory] = None,
        tools: Optional[ToolRegistry] = None,
        policy: Optional[AgentPolicy] = None,
        trace_factory: Optional[TraceFactory] = None,
        kb_description: str = "",
    ):
        self.config = config or load_config()
        self.kb = kb if kb is not None else create_knowledge_base(self.config.knowledge)
        self.memory = (
            memory
            if memory is not None
            else (AgentMemory(self.config.memory.db_path, enabled=self.config.memory.enabled) if self.config.memory.enabled else None)
        )
        self.llm = llm or create_llm(self.config.llm)
        self.tools = tools or build_tool_registry(self.config, self.kb, self.memory)
        self.policy = policy or AgentPolicy.from_config(self.config)
        runtime = AgentRuntime(
            llm=self.llm,
            tools=self.tools,
            config=self.config,
            kb=self.kb,
            memory=self.memory,
            policy=self.policy,
            trace_factory=trace_factory,
            kb_description=kb_description,
        )
        if self.config.agent.engine == "graph":
            import sys
            from pathlib import Path

            graphlibs = Path(__file__).resolve().parents[1] / ".agent" / "graphlibs"
            if graphlibs.exists():
                sys.path.insert(0, str(graphlibs))
            from .graph_runtime import GraphAgentRuntime

            self.runtime = GraphAgentRuntime(runtime)
        else:
            self.runtime = runtime

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_config(cls, path: Optional[str] = None, **kwargs: Any) -> "Agent":
        return cls(load_config(path), **kwargs)

    # -- 执行 -------------------------------------------------------------
    def ask(
        self,
        task: str,
        *,
        history: Optional[Sequence[Message]] = None,
        session_id: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
        persist: bool = True,
    ) -> AgentResult:
        """执行一次完整任务。

        Args:
            task: 用户任务/问题。
            history: 多轮历史；为 None 且提供 session_id 时自动从长期记忆恢复。
            session_id: 会话 ID，用于恢复历史与记录轮次。
            on_event: 事件回调（CLI 进度、SSE 流式）。
            persist: 是否把问答轮次写入长期记忆。
        """
        if not task or not task.strip():
            raise ValueError("task 不能为空")

        effective_history = history
        memory_context = ""
        if effective_history is None and session_id and self.memory is not None and self.memory.available:
            recent = self.memory.get_turns(session_id, limit=self.config.memory.history_turns * 2)
            effective_history = [Message(role=t["role"], content=t["content"]) for t in recent]
            memory_context = self._build_memory_context(session_id, task, recent)

        result = self.runtime.run(
            task,
            history=effective_history,
            session_id=session_id,
            on_event=on_event,
            persist_episode=persist,
            memory_context=memory_context,
        )

        if persist and session_id and self.memory is not None and self.memory.available:
            self.memory.append_turn(session_id, "user", task)
            self.memory.append_turn(session_id, "assistant", result.answer)
            self._maybe_update_summary(session_id)
        return result

    # -- 长期会话记忆 -----------------------------------------------------
    def _build_memory_context(
        self,
        session_id: str,
        task: str,
        recent: Sequence[dict],
    ) -> str:
        if self.memory is None or not self.memory.available:
            return ""
        summary = self.memory.get_summary(session_id)
        recent_ids = {item.get("id") for item in recent}
        related = self.memory.search_turns(
            session_id,
            task,
            limit=3,
            exclude_ids=recent_ids,
        )
        parts: List[str] = []
        if summary:
            parts.append("## 长期会话摘要\n" + summary.strip())
        if related:
            lines = []
            for item in related:
                role = "用户" if item.get("role") == "user" else "助手"
                content = str(item.get("content") or "").strip().replace("\n", " ")
                if content:
                    lines.append(f"- {role}：{content[:260]}")
            if lines:
                parts.append("## 相关历史对话\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def _maybe_update_summary(self, session_id: str) -> None:
        if self.memory is None or not self.memory.available:
            return
        turns = self.memory.get_turns(session_id, limit=200)
        keep = max(2, int(self.config.memory.history_turns) * 2)
        if len(turns) <= keep + 2:
            return
        old = turns[:-keep]
        existing = self.memory.get_summary(session_id)
        last_id = self.memory.get_summary_last_turn_id(session_id)
        old = [
            item for item in old
            if int(item.get("id") or 0) > last_id
        ]
        if not old:
            return
        lines: List[str] = []
        if existing:
            lines.append(existing.strip())
        for item in old:
            content = str(item.get("content") or "").strip().replace("\n", " ")
            if not content:
                continue
            role = "用户" if item.get("role") == "user" else "回答"
            limit = 140 if role == "用户" else 220
            lines.append(f"- {role}：{content[:limit]}")
        summary = "\n".join(lines).strip()
        if len(summary) > 1400:
            summary = summary[-1400:]
        if summary:
            self.memory.save_summary(session_id, summary, last_turn_id=int(old[-1].get("id") or 0))

    def stream(
        self,
        task: str,
        *,
        history: Optional[Sequence[Message]] = None,
        session_id: Optional[str] = None,
        persist: bool = True,
    ) -> Iterator[AgentEvent]:
        """以事件流方式执行任务（在后台线程运行，逐条 yield）。"""
        events: "queue.Queue[Optional[AgentEvent]]" = queue.Queue()
        error_box: List[BaseException] = []

        def on_event(event: AgentEvent) -> None:
            events.put(event)

        def worker() -> None:
            try:
                result = self.ask(
                    task,
                    history=history,
                    session_id=session_id,
                    on_event=on_event,
                    persist=persist,
                )
                # 完整结果作为最后一个事件下发，前端无需再发一次同步请求
                events.put(AgentEvent(type="result", payload=result.to_dict(include_steps=True)))
            except BaseException as exc:  # 把异常作为事件传出，避免生成器悬挂
                error_box.append(exc)
                events.put(
                    AgentEvent(
                        type="error",
                        payload={"error": str(exc), "type": type(exc).__name__},
                    )
                )
            finally:
                events.put(None)

        thread = threading.Thread(target=worker, name="agent-stream", daemon=True)
        thread.start()
        while True:
            item = events.get()
            if item is None:
                break
            yield item
        # 生成器结束前等待线程收尾（正常情况下已经结束）
        thread.join(timeout=1.0)

    # -- 诊断 / 资源 ------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self.runtime.health()

    @property
    def system_prompt(self) -> str:
        return self.runtime.system_prompt

    def close(self) -> None:
        try:
            self.runtime.close()
        except Exception:
            pass
        if self.memory is not None:
            try:
                self.memory.close()
            except Exception:
                pass
        if self.kb is not None:
            try:
                self.kb.close()
            except Exception:
                pass

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Agent"]
