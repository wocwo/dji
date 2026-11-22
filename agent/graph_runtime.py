"""LangGraph 版 ReAct 运行时（实验）。

把 AgentRuntime 的 while 循环拆成 4 个节点：

    prepare → think → act → think → ... → finalize

- think: 调用 LLM、解析工具调用/最终答案、证据闸门/事实覆盖检查
- act: 执行工具，把 observation 回灌
- finalize: 引用整理、grounding、claim、结果对象

legacy 运行时保持原样，通过 AGENT_ENGINE=graph 切换。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from .config import AgentConfig
from .grounding import extract_models, normalize_citations
from .parser import extract_citation_refs, parse_action
from .prompt import (
    format_duplicate_warning,
    format_evidence_required,
    format_invalid_action,
    format_missing_facts,
    format_no_evidence,
    format_observation,
    trim_history,
    truncate_text,
)
from .types import (
    AgentEvent,
    AgentResult,
    AgentStep,
    Message,
    ToolCall,
    Usage,
)
from .verifier import EvidenceGate


GraphState = Dict[str, Any]


class GraphAgentRuntime:
    """使用 LangGraph 节点执行 ReAct 循环。"""

    def __init__(self, legacy: Any):
        self.legacy = legacy
        self.config: AgentConfig = legacy.config
        self.policy = legacy.policy
        self.tools = legacy.tools
        self.llm = legacy.llm
        self.enabled_tools = legacy.enabled_tools
        self.evidence_gate: EvidenceGate = legacy.evidence_gate
        self.grounding_verifier = legacy.grounding_verifier
        self.claim_checker = legacy.claim_checker
        self.fact_checker = legacy.fact_checker
        self.memory = legacy.memory
        self._trace_factory = legacy._trace_factory

        graph = StateGraph(GraphState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("think", self._think)
        graph.add_node("act", self._act)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "think")
        graph.add_conditional_edges(
            "think",
            self._route_after_think,
            {"act": "act", "finalize": "finalize", "think": "think"},
        )
        graph.add_edge("act", "think")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    # -- 事件 -------------------------------------------------------------
    def _emit(self, state: GraphState, event_type: str, *, step: int = 0, **payload: Any) -> None:
        event = AgentEvent(type=event_type, step=step, payload=payload)
        state["tracer"].log(event_type, step=step, payload=payload)
        callback = state.get("on_event")
        if callback is not None:
            try:
                callback(event)
            except Exception:
                pass

    def _warn(self, state: GraphState, message: str, *, step: int = 0) -> None:
        warnings = state["warnings"]
        if message and message not in warnings:
            warnings.append(message)
            self._emit(state, "warning", step=step, message=message)

    def _mutate(self, state: GraphState, **updates: Any) -> GraphState:
        state.update(updates)
        return state

    # -- prepare ----------------------------------------------------------
    def _prepare(self, state: GraphState) -> GraphState:
        task = state["task"].strip()
        started_wall = time.time()
        budget = self.policy.budget()
        tracer = self._trace_factory()
        trace_id = tracer.start_run(
            task,
            {
                "provider": self.llm.name,
                "model": getattr(self.config.llm, "model", ""),
                "tool_mode": self.config.llm.tool_mode,
                "tools": self.enabled_tools,
                "kb": self.legacy.kb_description,
                "engine": "langgraph",
            },
        )
        from .tools.base import ToolContext

        ctx = ToolContext(
            run_id=trace_id,
            session_id=state.get("session_id"),
            memory=self.memory,
            kb=self.legacy.kb,
            policy=self.policy,
        )
        models = extract_models(task)
        ctx.requested_model = models[0] if models else ""
        profile = self.evidence_gate.profile(task)
        messages: List[Message] = [Message("system", self.system_prompt)]
        if state.get("memory_context"):
            messages.append(Message("system", state["memory_context"]))
        messages.extend(
            trim_history(
                state.get("history"),
                max_chars=self.config.agent.max_history_chars,
                max_turns=self.config.memory.history_turns,
            )
        )
        messages.append(Message("user", task))

        state.update(
            tracer=tracer,
            trace_id=trace_id,
            started_wall=started_wall,
            budget=budget,
        )

        self._emit(
            state,
            "run_start",
            task=task,
            provider=self.llm.name,
            model=getattr(self.config.llm, "model", ""),
            tool_mode=self.config.llm.tool_mode,
            tools=self.enabled_tools,
            kb=self.legacy.kb_description,
            session_id=state.get("session_id"),
            evidence_profile=profile.to_dict(),
            engine="langgraph",
        )

        # 快速模式预检索
        if (
            self.config.agent.prefetch_search
            and profile.requires_evidence
            and "manual_search" in self.enabled_tools
        ):
            call = ToolCall(
                name="manual_search",
                arguments={"query": task, "top_k": self.config.knowledge.top_k},
            )
            result = self.tools.execute(call, ctx, self.policy, enabled=self.enabled_tools)
            state["usage"].tool_calls += 1
            state["search_attempts"] += 1
            self.legacy._collect_evidence_texts(result, state["evidence_texts"])
            if self.legacy._is_evidence_result(call, result):
                state["evidence_count"] += 1
            step = AgentStep(
                index=len(state["steps"]) + 1,
                thought="快速模式：预检索手册",
                tool_call=call,
                result=result,
                latency_ms=result.latency_ms,
            )
            state["steps"].append(step)
            self._emit(state, "tool_call", step=step.index, name=call.name, arguments=call.arguments)
            self._emit(
                state,
                "tool_result",
                step=step.index,
                name=call.name,
                ok=result.ok,
                content=truncate_text(result.content, 1200),
                error=result.error,
                citations=[c.ref for c in result.citations],
            )
            messages.append(
                Message(
                    "user",
                    "下面是系统预检索到的手册证据。请直接基于它回答；"
                    "如果证据足够，不要再调用 manual_search。\n\n"
                    + format_observation(result, max_chars=self.policy.max_observation_chars),
                )
            )

        return self._mutate(
            state,
            started_wall=started_wall,
            budget=budget,
            tracer=tracer,
            trace_id=trace_id,
            ctx=ctx,
            profile=profile,
            messages=messages,
            final_emitted=False,
            llm_failed=False,
            forced_final=False,
        )

    # -- think ------------------------------------------------------------
    def _think(self, state: GraphState) -> GraphState:
        budget = state["budget"]
        reason = budget.reason(
            steps=len(state["steps"]),
            tool_calls=state["usage"].tool_calls,
            llm_calls=state["usage"].llm_calls,
        )
        if reason:
            state["status"] = reason
            state["forced_final"] = True
            self._emit(state, "budget", step=len(state["steps"]) + 1, reason=reason, **budget.to_dict())
            return self._mutate(state, next_action="finalize")

        resp = self.legacy._complete(
            state["messages"],
            use_tools=True,
            warnings=state["warnings"],
        )
        if resp is None:
            state["status"] = "error"
            state["llm_failed"] = True
            self._warn(state, "LLM 调用失败，无法继续推理", step=len(state["steps"]) + 1)
            return self._mutate(state, next_action="finalize")

        state["usage"].merge(resp.usage)
        state["usage"].llm_calls += 1
        self._emit(
            state,
            "llm_response",
            step=len(state["steps"]) + 1,
            text=truncate_text(resp.text, 2000),
            tool_calls=[c.to_dict() for c in resp.tool_calls],
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model=resp.model,
        )

        if resp.tool_calls:
            state["messages"].append(Message("assistant", resp.text or "", tool_calls=resp.tool_calls))
            return self._mutate(state, pending_calls=list(resp.tool_calls), pending_text_call=None, next_action="act")

        allow_plain_final = self.config.agent.plain_text_final or self.config.llm.tool_mode == "native"
        parsed = parse_action(resp.text, plain_text_final=allow_plain_final)

        if parsed.kind == "final":
            text = normalize_citations(parsed.final_answer.strip())
            if not text:
                step = AgentStep(
                    index=len(state["steps"]) + 1,
                    thought=parsed.thought,
                    raw_response=resp.text,
                    error="最终答案为空",
                )
                state["steps"].append(step)
                state["messages"].append(Message("assistant", resp.text))
                state["messages"].append(Message("user", format_invalid_action("final_answer 为空")))
                state["parse_error_streak"] += 1
                self._warn(state, "模型返回了空的最终答案", step=step.index)
                action = "finalize" if state["parse_error_streak"] >= self.policy.max_consecutive_errors else "think"
                if action == "finalize":
                    state["status"] = "error"
                return self._mutate(state, next_action=action)

            check = self.evidence_gate.check(
                state["profile"],
                text,
                known_refs=[c.ref for c in state["ctx"].all_citations()],
                evidence_count=state["evidence_count"],
            )
            if check.needs_retrieval and state["evidence_retries"] < int(self.config.agent.max_evidence_retries):
                state["evidence_retries"] += 1
                rejected = AgentStep(
                    index=len(state["steps"]) + 1,
                    thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                    raw_response=resp.text,
                    error="缺少手册证据",
                )
                state["steps"].append(rejected)
                state["messages"].append(Message("assistant", resp.text))
                self._warn(state, "回答缺少手册证据，触发 RAG 回退", step=rejected.index)
                can_auto_search = (
                    self.evidence_gate.auto_search
                    and state["search_attempts"] == 0
                    and "manual_search" in self.enabled_tools
                    and state["usage"].tool_calls < self.policy.max_tool_calls
                    and budget.elapsed() < budget.max_seconds
                )
                if can_auto_search:
                    call = ToolCall(
                        name="manual_search",
                        arguments={"query": state["task"], "top_k": self.config.knowledge.top_k},
                    )
                    return self._mutate(state, pending_calls=[], pending_text_call=call, auto_search=True, next_action="act")
                state["messages"].append(
                    Message(
                        "user",
                        format_no_evidence() if state["search_attempts"] > 0 else format_evidence_required(),
                    )
                )
                return self._mutate(state, next_action="think")

            if check.needs_retrieval and self.evidence_gate.enabled:
                self._warn(state, "最终答案未获得手册证据（自动检索未成功或已关闭）", step=len(state["steps"]) + 1)

            if self.fact_checker is not None and state["ctx"].requested_model:
                report = self.fact_checker.check(state["task"], text)
                state["fact_coverage_report"] = report
                if report.missing:
                    if state["fact_retries"] < int(self.config.agent.max_fact_retries):
                        state["fact_retries"] += 1
                        rejected = AgentStep(
                            index=len(state["steps"]) + 1,
                            thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                            raw_response=resp.text,
                            error="关键事实未覆盖",
                        )
                        state["steps"].append(rejected)
                        state["messages"].append(Message("assistant", resp.text))
                        state["messages"].append(Message("user", format_missing_facts(report.missing)))
                        self._warn(
                            state,
                            "回答缺少关键事实：" + "、".join(report.missing) + "，触发一次补答",
                            step=rejected.index,
                        )
                        return self._mutate(state, next_action="think")
                    self._warn(
                        state,
                        "关键事实仍未覆盖：" + "、".join(report.missing),
                        step=len(state["steps"]) + 1,
                    )

            step = AgentStep(
                index=len(state["steps"]) + 1,
                thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                raw_response=resp.text,
            )
            state["steps"].append(step)
            state["answer"] = text
            state["status"] = "ok"
            state["final_emitted"] = True
            self._emit(state, "final", step=step.index, answer=text, soft=parsed.soft)
            return self._mutate(state, next_action="finalize")

        if parsed.kind == "invalid":
            step = AgentStep(
                index=len(state["steps"]) + 1,
                raw_response=resp.text,
                error=parsed.error or "解析失败",
            )
            state["steps"].append(step)
            state["messages"].append(Message("assistant", resp.text))
            state["messages"].append(Message("user", format_invalid_action(step.error or "格式错误")))
            state["parse_error_streak"] += 1
            self._warn(state, f"动作解析失败：{step.error}", step=step.index)
            action = "finalize" if state["parse_error_streak"] >= self.policy.max_consecutive_errors else "think"
            if action == "finalize":
                state["status"] = "error"
            return self._mutate(state, next_action=action)

        state["pending_text_call"] = parsed.tool_call
        state["pending_calls"] = []
        state["auto_search"] = False
        return self._mutate(state, next_action="act")

    # -- act --------------------------------------------------------------
    def _act(self, state: GraphState) -> GraphState:
        budget = state["budget"]
        pending: List[ToolCall] = state.get("pending_calls") or []
        text_call: Optional[ToolCall] = state.get("pending_text_call")
        if pending:
            if state["usage"].tool_calls + len(pending) > self.policy.max_tool_calls:
                state["status"] = "max_tool_calls"
                state["forced_final"] = True
                return self._mutate(state, next_action="finalize")
        elif text_call is not None:
            pending = [text_call]

        for call in pending:
            signature = (
                call.name,
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str),
            )
            if signature in state["duplicate_seen"]:
                step = AgentStep(
                    index=len(state["steps"]) + 1,
                    thought="",
                    tool_call=call,
                    error="重复调用",
                )
                state["steps"].append(step)
                state["messages"].append(Message("user", format_duplicate_warning(call.name)))
                self._warn(state, f"拦截重复工具调用：{call.name}", step=step.index)
                continue
            state["duplicate_seen"].add(signature)
            result = self.tools.execute(call, state["ctx"], self.policy, enabled=self.enabled_tools)
            state["usage"].tool_calls += 1
            self.legacy._collect_evidence_texts(result, state["evidence_texts"])
            if call.name == "manual_search":
                state["search_attempts"] += 1
            if self.legacy._is_evidence_result(call, result):
                state["evidence_count"] += 1
            step = AgentStep(
                index=len(state["steps"]) + 1,
                thought="",
                tool_call=call,
                result=result,
                latency_ms=result.latency_ms,
            )
            state["steps"].append(step)
            self._emit(state, "tool_call", step=step.index, name=call.name, arguments=call.arguments)
            self._emit(
                state,
                "tool_result",
                step=step.index,
                name=call.name,
                ok=result.ok,
                content=truncate_text(result.content, 1200),
                error=result.error,
                citations=[c.ref for c in result.citations],
            )
            if state.get("auto_search"):
                state["messages"].append(
                    Message(
                        "user",
                        format_observation(result, max_chars=self.policy.max_observation_chars),
                    )
                )
            elif state.get("pending_calls"):
                state["messages"].append(
                    Message(
                        "tool",
                        format_observation(result, max_chars=self.policy.max_observation_chars),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            else:
                state["messages"].append(Message("assistant", ""))
                state["messages"].append(
                    Message(
                        "user",
                        format_observation(result, max_chars=self.policy.max_observation_chars),
                    )
                )

        return self._mutate(state, pending_calls=[], pending_text_call=None, auto_search=False, next_action="think")

    def _route_after_think(self, state: GraphState) -> str:
        return state.get("next_action", "finalize")

    # -- finalize ---------------------------------------------------------
    def _finalize(self, state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        status = state.get("status", "ok")
        forced_final = state.get("forced_final", False)
        usage = state["usage"]
        warnings = state["warnings"]
        steps = state["steps"]
        messages = state["messages"]
        task = state["task"]
        ctx = state["ctx"]
        profile = state["profile"]
        budget = state["budget"]
        tracer = state["tracer"]

        def emit(event_type: str, *, step: int = 0, **payload: Any) -> None:
            self._emit(state, event_type, step=step, **payload)

        def warn(message: str, *, step: int = 0) -> None:
            self._warn(state, message, step=step)

        if not answer:
            if status == "ok":
                status = "no_answer"
            if not state.get("llm_failed") and status != "error" and budget.elapsed() < budget.max_seconds:
                forced_final = True
                answer = self.legacy._force_final(
                    messages,
                    usage=usage,
                    warnings=warnings,
                    emit=emit,
                    warn=warn,
                    status=status,
                ) or ""
            if not answer:
                answer = self.legacy._fallback_answer(steps, status)
                if not warnings:
                    warn("未能生成最终答案，已返回已检索到的信息摘要")

        answer = normalize_citations(answer)
        all_citations = ctx.all_citations()
        refs = extract_citation_refs(answer)
        if refs:
            known = {c.ref: c for c in all_citations}
            unknown = [ref for ref in refs if ref not in known]
            if unknown:
                warn(f"最终答案引用了不存在的编号：{', '.join(unknown)}")
            citations = [known[ref] for ref in refs if ref in known]
        else:
            citations = all_citations
            if all_citations:
                warn("最终答案未标注引用编号")

        grounding = None
        if self.config.agent.grounding_check and (profile.requires_evidence or state["evidence_count"] > 0 or all_citations):
            grounding = self.grounding_verifier.check(
                answer,
                all_citations,
                state["evidence_texts"],
                question=task,
            )
            if grounding.issues:
                warn(
                    f"引用校验：支持度 {grounding.score:.0%}，引用覆盖 {grounding.coverage:.0%}，"
                    f"{len(grounding.issues)} 个问题",
                    step=len(steps),
                )
                for issue in grounding.issues[:3]:
                    if issue.kind in ("numeric_mismatch", "model_scope", "malformed", "uncited"):
                        warn(f"引用校验：{issue.detail}", step=len(steps))

        claim_report = None
        claim_gate = not self.config.agent.claim_verifier_high_risk_only
        if not claim_gate:
            claim_gate = getattr(profile, "risk", "") == "high"
        if not claim_gate and grounding is not None:
            claim_gate = any(issue.kind in ("numeric_mismatch", "model_scope") for issue in grounding.issues)
        if self.claim_checker is not None and profile.requires_evidence and all_citations and answer and claim_gate:
            claim_report = self.claim_checker.check(task, answer, all_citations, state["evidence_texts"])
            usage.merge(claim_report.usage)
            usage.llm_calls += 1
            if claim_report.error:
                warn(f"语义复核失败：{claim_report.error}", step=len(steps))
            elif claim_report.issues:
                warn(f"语义复核：{len(claim_report.issues)} 条 claim 需要复核", step=len(steps))

        duration_ms = (time.time() - state["started_wall"]) * 1000
        if answer:
            messages.append(Message("assistant", answer.strip()))
        if not state.get("final_emitted"):
            emit("final", step=len(steps), answer=answer, forced=forced_final, status=status)
        result = AgentResult(
            task=task,
            answer=answer.strip(),
            status=status,
            steps=steps,
            citations=citations,
            usage=usage,
            trace_id=state["trace_id"],
            duration_ms=duration_ms,
            warnings=warnings,
            forced_final=forced_final,
            metadata={
                "provider": self.llm.name,
                "model": getattr(self.config.llm, "model", ""),
                "tool_mode": self.config.llm.tool_mode,
                "kb": self.legacy.kb_description,
                "session_id": state.get("session_id"),
                "plan": list(ctx.plan),
                "budget": budget.to_dict(),
                "evidence_profile": profile.to_dict(),
                "requested_model": ctx.requested_model,
                "evidence_count": state["evidence_count"],
                "evidence_retries": state["evidence_retries"],
                "grounding": grounding.to_dict() if grounding is not None else None,
                "claim_verification": claim_report.to_dict() if claim_report is not None else None,
                "citation_alignment": None,
                "fact_coverage": (
                    state["fact_coverage_report"].to_dict()
                    if state.get("fact_coverage_report") is not None
                    else None
                ),
                "config_path": self.config.config_path,
                "trace_path": str(tracer.path) if tracer.path else "",
                "engine": "langgraph",
            },
            transcript=messages,
        )
        if self.memory is not None and state.get("persist_episode", True):
            try:
                episode_id = self.memory.save_episode(result, session_id=state.get("session_id"))
                if episode_id:
                    result.metadata["episode_id"] = episode_id
            except Exception:
                pass
        tracer.finish(result)
        emit(
            "run_end",
            status=status,
            answer=answer,
            citations=[c.ref for c in citations],
            duration_ms=round(duration_ms, 1),
            steps=len(steps),
            tool_calls=usage.tool_calls,
            forced_final=forced_final,
            warnings=warnings,
        )
        return self._mutate(state, result=result)

    # -- public API -------------------------------------------------------
    def run(
        self,
        task: str,
        *,
        history: Optional[Any] = None,
        session_id: Optional[str] = None,
        on_event: Optional[Any] = None,
        persist_episode: bool = True,
        memory_context: str = "",
    ) -> AgentResult:
        state = dict(
            task=task,
            history=history,
            session_id=session_id,
            on_event=on_event,
            persist_episode=persist_episode,
            memory_context=memory_context,
            usage=Usage(),
            steps=[],
            warnings=[],
            status="ok",
            answer="",
            forced_final=False,
            final_emitted=False,
            llm_failed=False,
            duplicate_seen=set(),
            parse_error_streak=0,
            evidence_count=0,
            evidence_retries=0,
            search_attempts=0,
            fact_retries=0,
            fact_coverage_report=None,
            evidence_texts={},
            pending_calls=[],
            pending_text_call=None,
            auto_search=False,
            next_action="think",
        )
        result = self.graph.invoke(state)
        return result["result"]

    @property
    def system_prompt(self) -> str:
        return self.legacy.system_prompt

    def health(self) -> Dict[str, Any]:
        result = dict(self.legacy.health())
        result["engine"] = "langgraph"
        result["nodes"] = ["prepare", "think", "act", "finalize"]
        return result

    def close(self) -> None:
        self.legacy.close()


__all__ = ["GraphAgentRuntime", "GraphState"]
