"""Agent 运行时：ReAct 风格的多步「思考—行动—观察」循环。

职责：
- 维护工作记忆（messages）与全局引用编号；
- 调用 LLM，解析工具调用或最终答案；
- 执行工具、把结果作为 observation 回灌；
- 预算控制、错误恢复、重复调用拦截；
- 产出结构化 :class:`AgentResult`，并写入 trace 与长期记忆。

它不绑定具体 LLM、工具或知识库，便于替换与测试。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import AgentConfig
from .claim_verifier import ClaimVerifier
from .citation_aligner import CitationAligner
from .fact_coverage import FactCoverageChecker
from .grounding import GroundingVerifier, extract_models, normalize_citations
from .llm import LLMClient, LLMResponse
from .parser import ParsedAction, extract_citation_refs, parse_action
from .policy import AgentPolicy, Budget
from .prompt import (
    build_system_prompt,
    format_duplicate_warning,
    format_evidence_required,
    format_forced_final,
    format_invalid_action,
    format_missing_facts,
    format_no_evidence,
    format_observation,
    trim_history,
    truncate_text,
)
from .tools.base import ToolContext, ToolRegistry
from .trace import Tracer
from .types import (
    AgentEvent,
    AgentResult,
    AgentStep,
    Citation,
    Message,
    ToolCall,
    Usage,
    now_iso,
)
from .verifier import EvidenceGate

logger = logging.getLogger(__name__)

_TROUBLESHOOT_TERMS = (
    "坠机",
    "摔",
    "掉下来",
    "炸机",
    "无法开机",
    "开不了机",
    "不能开机",
    "进水",
    "撞击",
    "坠落",
    "迫降",
)


def _is_troubleshooting(task: str) -> bool:
    return any(term in (task or "") for term in _TROUBLESHOOT_TERMS)


EventCallback = Callable[[AgentEvent], None]
TraceFactory = Callable[[], Tracer]


class AgentRuntime:
    """Agent 运行循环。"""

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
        config: AgentConfig,
        kb: Any = None,
        memory: Any = None,
        policy: Optional[AgentPolicy] = None,
        trace_factory: Optional[TraceFactory] = None,
        kb_description: str = "",
    ):
        self.llm = llm
        self.tools = tools
        self.config = config
        self.kb = kb
        self.memory = memory
        self.policy = policy or AgentPolicy.from_config(config)
        self.evidence_gate = EvidenceGate(
            enabled=config.agent.evidence_guard,
            auto_search=config.agent.evidence_auto_search,
        )
        self.grounding_verifier = GroundingVerifier()
        self.claim_checker = (
            ClaimVerifier(
                self.llm,
                max_claims=config.agent.claim_verifier_max_claims,
            )
            if config.agent.claim_verifier
            else None
        )
        self.citation_aligner = (
            CitationAligner() if config.agent.citation_align else None
        )
        self.fact_checker = (
            FactCoverageChecker() if config.agent.fact_coverage else None
        )
        self._trace_factory: TraceFactory = trace_factory or (
            lambda: Tracer(
                config.trace.dir,
                enabled=config.trace.enabled,
                include_llm_output=config.trace.include_llm_output,
            )
        )

        self.enabled_tools: List[str] = [name for name in config.tools.enabled if name in tools.names()]
        kb_desc = kb_description or (getattr(kb, "description", "") if kb is not None else "（未配置知识库）")
        self.kb_description = kb_desc
        self.system_prompt = build_system_prompt(
            tools.specs(self.enabled_tools),
            agent_name=config.agent.name,
            kb_description=kb_desc,
            tool_mode=config.llm.tool_mode,
            max_steps=config.agent.max_steps,
        )

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(
        self,
        task: str,
        *,
        history: Optional[Sequence[Message]] = None,
        session_id: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
        persist_episode: bool = True,
        memory_context: str = "",
    ) -> AgentResult:
        task = (task or "").strip()
        if not task:
            raise ValueError("task 不能为空")

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
                "kb": self.kb_description,
            },
        )

        ctx = ToolContext(
            run_id=trace_id,
            session_id=session_id,
            memory=self.memory,
            kb=self.kb,
            policy=self.policy,
        )
        requested_models = extract_models(task)
        ctx.requested_model = requested_models[0] if requested_models else ""
        usage = Usage()
        steps: List[AgentStep] = []
        warnings: List[str] = []
        status = "ok"
        answer = ""
        forced_final = False
        final_emitted = False
        llm_failed = False
        duplicate_seen: set = set()
        parse_error_streak = 0
        profile = self.evidence_gate.profile(task)
        evidence_count = 0
        evidence_retries = 0
        search_attempts = 0
        fact_retries = 0
        fact_coverage_report = None
        general_fallback = _is_troubleshooting(task)
        general_fallback_mode = False
        evidence_texts: Dict[str, str] = {}

        def emit(event_type: str, *, step: int = 0, **payload: Any) -> None:
            event = AgentEvent(type=event_type, step=step, payload=payload)
            tracer.log(event_type, step=step, payload=payload)
            if on_event is not None:
                try:
                    on_event(event)
                except Exception:  # 回调不应影响主流程
                    logger.exception("on_event 回调异常")

        def warn(message: str, *, step: int = 0) -> None:
            if message and message not in warnings:
                warnings.append(message)
                emit("warning", step=step, message=message)

        emit(
            "run_start",
            task=task,
            provider=self.llm.name,
            model=getattr(self.config.llm, "model", ""),
            tool_mode=self.config.llm.tool_mode,
            tools=self.enabled_tools,
            kb=self.kb_description,
            session_id=session_id,
            evidence_profile=profile.to_dict(),
        )

        messages: List[Message] = [Message("system", self.system_prompt)]
        if memory_context:
            messages.append(Message("system", memory_context))
        messages.extend(
            trim_history(
                history,
                max_chars=self.config.agent.max_history_chars,
                max_turns=self.config.memory.history_turns,
            )
        )
        messages.append(Message("user", task))
        if general_fallback:
            messages.append(
                Message(
                    "system",
                    "故障排查例外：当用户描述坠机、摔落、进水、无法开机等硬件故障时，"
                    "允许给出通用排查建议，但必须在对应段落标注「通用建议（非手册）」；"
                    "手册没有覆盖的部分明确说「手册未覆盖」，禁止编造手册、页码或官方结论。"
                    "安全优先：不要建议拆机，不要给鼓包/破损/进水的电池充电。",
                )
            )

        if (
            self.config.agent.prefetch_search
            and profile.requires_evidence
            and "manual_search" in self.enabled_tools
        ):
            prefetch_call = ToolCall(
                name="manual_search",
                arguments={"query": task, "top_k": self.config.knowledge.top_k},
            )
            prefetch_result = self.tools.execute(
                prefetch_call, ctx, self.policy, enabled=self.enabled_tools
            )
            usage.tool_calls += 1
            search_attempts += 1
            self._collect_evidence_texts(prefetch_result, evidence_texts)
            if self._is_evidence_result(prefetch_call, prefetch_result):
                evidence_count += 1
            prefetch_step = AgentStep(
                index=len(steps) + 1,
                thought="快速模式：预检索手册",
                tool_call=prefetch_call,
                result=prefetch_result,
                latency_ms=prefetch_result.latency_ms,
            )
            steps.append(prefetch_step)
            emit("tool_call", step=prefetch_step.index, name=prefetch_call.name, arguments=prefetch_call.arguments)
            emit(
                "tool_result",
                step=prefetch_step.index,
                name=prefetch_call.name,
                ok=prefetch_result.ok,
                content=truncate_text(prefetch_result.content, 1200),
                error=prefetch_result.error,
                citations=[c.ref for c in prefetch_result.citations],
            )
            messages.append(
                Message(
                    "user",
                    "下面是系统预检索到的手册证据。请直接基于它回答；"
                    "如果证据足够，不要再调用 manual_search。\n\n"
                    + format_observation(
                        prefetch_result,
                        max_chars=self.policy.max_observation_chars,
                    ),
                )
            )

        while True:
            reason = budget.reason(
                steps=len(steps),
                tool_calls=usage.tool_calls,
                llm_calls=usage.llm_calls,
            )
            if reason:
                status = reason
                forced_final = True
                emit("budget", step=len(steps) + 1, reason=reason, **budget.to_dict())
                break

            resp = self._complete(messages, use_tools=True, warnings=warnings)
            if resp is None:
                status = "error"
                llm_failed = True
                warn("LLM 调用失败，无法继续推理", step=len(steps) + 1)
                break

            usage.merge(resp.usage)
            usage.llm_calls += 1
            emit(
                "llm_response",
                step=len(steps) + 1,
                text=truncate_text(resp.text, 2000),
                tool_calls=[c.to_dict() for c in resp.tool_calls],
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                model=resp.model,
            )

            # ---- 原生 function calling --------------------------------
            if resp.tool_calls:
                messages.append(Message("assistant", resp.text or "", tool_calls=resp.tool_calls))
                stop = False
                for call_index, call in enumerate(resp.tool_calls):
                    reason = budget.reason(
                        steps=len(steps) + 1,
                        tool_calls=usage.tool_calls,
                        llm_calls=usage.llm_calls,
                    )
                    if reason:
                        status = reason
                        forced_final = True
                        stop = True
                        emit("budget", step=len(steps) + 1, reason=reason)
                        # 为未执行的调用补上 tool 消息，保持原生协议完整
                        for remaining in resp.tool_calls[call_index:]:
                            messages.append(
                                Message(
                                    "tool",
                                    "（预算耗尽，未执行）",
                                    tool_call_id=remaining.id,
                                    name=remaining.name,
                                )
                            )
                        break

                    result = self.tools.execute(call, ctx, self.policy, enabled=self.enabled_tools)
                    usage.tool_calls += 1
                    self._collect_evidence_texts(result, evidence_texts)
                    if call.name == "manual_search":
                        search_attempts += 1
                    if self._is_evidence_result(call, result):
                        evidence_count += 1
                    step = AgentStep(
                        index=len(steps) + 1,
                        thought=(resp.text or "").strip(),
                        tool_call=call,
                        result=result,
                        raw_response=resp.text or "",
                        latency_ms=result.latency_ms,
                    )
                    steps.append(step)
                    emit("tool_call", step=step.index, name=call.name, arguments=call.arguments)
                    emit(
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
                            "tool",
                            format_observation(result, max_chars=self.policy.max_observation_chars),
                            tool_call_id=call.id,
                            name=call.name,
                        )
                    )
                    if not result.ok and result.error:
                        logger.info("工具 %s 失败: %s", call.name, result.error)

                if stop:
                    break
                continue

            # ---- 文本协议 --------------------------------------------
            # native 模式下模型通常直接返回自然语言答案，不需要 JSON 协议
            allow_plain_final = (
                self.config.agent.plain_text_final
                or self.config.llm.tool_mode == "native"
            )
            parsed = parse_action(
                resp.text,
                plain_text_final=allow_plain_final,
            )

            if parsed.kind == "final":
                text = normalize_citations(parsed.final_answer.strip())
                if not text:
                    step = AgentStep(
                        index=len(steps) + 1,
                        thought=parsed.thought,
                        raw_response=resp.text,
                        error="最终答案为空",
                    )
                    steps.append(step)
                    messages.append(Message("assistant", resp.text))
                    messages.append(Message("user", format_invalid_action("final_answer 为空")))
                    parse_error_streak += 1
                    warn("模型返回了空的最终答案", step=step.index)
                    if parse_error_streak >= self.policy.max_consecutive_errors:
                        status = "error"
                        break
                    continue

                check = None
                if not (general_fallback and search_attempts > 0):
                    check = self.evidence_gate.check(
                        profile,
                        text,
                        known_refs=[citation.ref for citation in ctx.all_citations()],
                        evidence_count=evidence_count,
                    )
                if (
                    check is not None
                    and check.needs_retrieval
                    and evidence_retries < int(self.config.agent.max_evidence_retries)
                ):
                    evidence_retries += 1
                    rejected = AgentStep(
                        index=len(steps) + 1,
                        thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                        raw_response=resp.text,
                        error="缺少手册证据",
                    )
                    steps.append(rejected)
                    messages.append(Message("assistant", resp.text))
                    warn("回答缺少手册证据，触发 RAG 回退", step=rejected.index)
                    auto_search_done = False
                    can_auto_search = (
                        self.evidence_gate.auto_search
                        and search_attempts == 0
                        and "manual_search" in self.enabled_tools
                        and usage.tool_calls < self.policy.max_tool_calls
                        and budget.elapsed() < budget.max_seconds
                    )
                    if can_auto_search:
                        auto_call = ToolCall(
                            name="manual_search",
                            arguments={
                                "query": task,
                                "top_k": self.config.knowledge.top_k,
                            },
                        )
                        auto_result = self.tools.execute(
                            auto_call,
                            ctx,
                            self.policy,
                            enabled=self.enabled_tools,
                        )
                        usage.tool_calls += 1
                        search_attempts += 1
                        self._collect_evidence_texts(auto_result, evidence_texts)
                        duplicate_seen.add(
                            (
                                auto_call.name,
                                json.dumps(
                                    auto_call.arguments,
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    default=str,
                                ),
                            )
                        )
                        if self._is_evidence_result(auto_call, auto_result):
                            evidence_count += 1
                        auto_step = AgentStep(
                            index=len(steps) + 1,
                            thought="证据闸门：自动检索手册",
                            tool_call=auto_call,
                            result=auto_result,
                            latency_ms=auto_result.latency_ms,
                        )
                        steps.append(auto_step)
                        emit(
                            "tool_call",
                            step=auto_step.index,
                            name=auto_call.name,
                            arguments=auto_call.arguments,
                            auto=True,
                        )
                        emit(
                            "tool_result",
                            step=auto_step.index,
                            name=auto_call.name,
                            ok=auto_result.ok,
                            content=truncate_text(auto_result.content, 1200),
                            error=auto_result.error,
                            citations=[c.ref for c in auto_result.citations],
                            auto=True,
                        )
                        messages.append(
                            Message(
                                "user",
                                format_observation(
                                    auto_result,
                                    max_chars=self.policy.max_observation_chars,
                                ),
                            )
                        )
                        auto_search_done = True
                    if general_fallback:
                        general_fallback_mode = True
                        messages.append(
                            Message(
                                "user",
                                "手册没有覆盖这个硬件故障场景。请按「故障排查例外」回答："
                                "给通用排查建议并标注「通用建议（非手册）」，先保证安全；"
                                "不要编造手册页码或官方结论。",
                            )
                        )
                    elif not auto_search_done:
                        messages.append(
                            Message(
                                "user",
                                format_no_evidence()
                                if search_attempts > 0
                                else format_evidence_required(),
                            )
                        )
                    continue

                if check is not None and check.needs_retrieval and self.evidence_gate.enabled:
                    warn(
                        "最终答案未获得手册证据（自动检索未成功或已关闭）",
                        step=len(steps) + 1,
                    )

                if self.fact_checker is not None and ctx.requested_model:
                    fact_coverage_report = self.fact_checker.check(task, text)
                    if fact_coverage_report.missing:
                        if fact_retries < int(self.config.agent.max_fact_retries):
                            fact_retries += 1
                            rejected = AgentStep(
                                index=len(steps) + 1,
                                thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                                raw_response=resp.text,
                                error="关键事实未覆盖",
                            )
                            steps.append(rejected)
                            messages.append(Message("assistant", resp.text))
                            messages.append(
                                Message(
                                    "user",
                                    format_missing_facts(fact_coverage_report.missing),
                                )
                            )
                            warn(
                                "回答缺少关键事实："
                                + "、".join(fact_coverage_report.missing)
                                + "，触发一次补答",
                                step=rejected.index,
                            )
                            continue
                        warn(
                            "关键事实仍未覆盖："
                            + "、".join(fact_coverage_report.missing),
                            step=len(steps) + 1,
                        )

                step = AgentStep(
                    index=len(steps) + 1,
                    thought=parsed.thought or ("（纯文本直答）" if parsed.soft else ""),
                    raw_response=resp.text,
                )
                steps.append(step)
                answer = text
                status = "ok"
                if parsed.soft:
                    if self.config.llm.tool_mode != "native":
                        warn(
                            "模型未按 JSON 协议输出，已把纯文本作为最终答案",
                            step=step.index,
                        )
                emit("final", step=step.index, answer=answer, soft=parsed.soft)
                final_emitted = True
                break

            if parsed.kind == "invalid":
                step = AgentStep(
                    index=len(steps) + 1,
                    raw_response=resp.text,
                    error=parsed.error or "解析失败",
                )
                steps.append(step)
                messages.append(Message("assistant", resp.text))
                messages.append(Message("user", format_invalid_action(step.error or "格式错误")))
                parse_error_streak += 1
                warn(f"动作解析失败：{step.error}", step=step.index)
                if parse_error_streak >= self.policy.max_consecutive_errors:
                    status = "error"
                    warn("连续解析失败次数过多，提前结束", step=step.index)
                    break
                continue

            # ---- 工具调用 --------------------------------------------
            parse_error_streak = 0
            assert parsed.tool_call is not None
            call = parsed.tool_call
            signature = (
                call.name,
                json.dumps(call.arguments, ensure_ascii=False, sort_keys=True, default=str),
            )
            if signature in duplicate_seen:
                step = AgentStep(
                    index=len(steps) + 1,
                    thought=parsed.thought,
                    tool_call=call,
                    raw_response=resp.text,
                    error="重复调用",
                )
                steps.append(step)
                messages.append(Message("assistant", resp.text))
                messages.append(Message("user", format_duplicate_warning(call.name)))
                warn(f"拦截重复工具调用：{call.name}", step=step.index)
                continue
            duplicate_seen.add(signature)

            result = self.tools.execute(call, ctx, self.policy, enabled=self.enabled_tools)
            usage.tool_calls += 1
            self._collect_evidence_texts(result, evidence_texts)
            if call.name == "manual_search":
                search_attempts += 1
            if self._is_evidence_result(call, result):
                evidence_count += 1
            step = AgentStep(
                index=len(steps) + 1,
                thought=parsed.thought,
                tool_call=call,
                result=result,
                raw_response=resp.text,
                latency_ms=result.latency_ms,
            )
            steps.append(step)
            emit("tool_call", step=step.index, name=call.name, arguments=call.arguments)
            emit(
                "tool_result",
                step=step.index,
                name=call.name,
                ok=result.ok,
                content=truncate_text(result.content, 1200),
                error=result.error,
                citations=[c.ref for c in result.citations],
            )
            messages.append(Message("assistant", resp.text))
            messages.append(
                Message(
                    "user",
                    format_observation(result, max_chars=self.policy.max_observation_chars),
                )
            )

        # ------------------------------------------------------------------
        # 收尾：强制生成 / 兜底摘要
        # ------------------------------------------------------------------
        if not answer:
            if status == "ok":
                status = "no_answer"
            if not llm_failed and status != "error" and budget.elapsed() < budget.max_seconds:
                forced_final = True
                answer = self._force_final(
                    messages,
                    usage=usage,
                    warnings=warnings,
                    emit=emit,
                    warn=warn,
                    status=status,
                ) or ""
            if not answer:
                answer = self._fallback_answer(steps, status)
                if not warnings:
                    warn("未能生成最终答案，已返回已检索到的信息摘要")

        answer = normalize_citations(answer)

        # ------------------------------------------------------------------
        # 引用与结果
        # ------------------------------------------------------------------
        all_citations = ctx.all_citations()
        citation_alignment = None
        if self.citation_aligner is not None and answer and all_citations:
            citation_alignment = self.citation_aligner.align(
                answer,
                all_citations,
                evidence_texts,
            )
            if citation_alignment.changes:
                answer = citation_alignment.answer
                warn(
                    f"引用页对齐：调整了 {len(citation_alignment.changes)} 处引用",
                    step=len(steps),
                )

        refs = extract_citation_refs(answer)
        if refs:
            known = {c.ref: c for c in all_citations}
            unknown = [ref for ref in refs if ref not in known]
            if unknown:
                warn(f"最终答案引用了不存在的编号：{', '.join(unknown)}")
            citations = [known[ref] for ref in refs if ref in known]
        else:
            if general_fallback_mode:
                citations = []
            else:
                citations = all_citations
                if all_citations:
                    warn("最终答案未标注引用编号")

        grounding = None
        if (
            self.config.agent.grounding_check
            and (profile.requires_evidence or evidence_count > 0 or all_citations)
        ):
            grounding = self.grounding_verifier.check(
                answer,
                all_citations,
                evidence_texts,
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
            claim_gate = any(
                issue.kind in ("numeric_mismatch", "model_scope")
                for issue in grounding.issues
            )
        if (
            self.claim_checker is not None
            and profile.requires_evidence
            and all_citations
            and answer
            and claim_gate
        ):
            claim_report = self.claim_checker.check(
                task,
                answer,
                all_citations,
                evidence_texts,
            )
            usage.merge(claim_report.usage)
            usage.llm_calls += 1
            if claim_report.error:
                warn(f"语义复核失败：{claim_report.error}", step=len(steps))
            elif claim_report.issues:
                warn(
                    f"语义复核：{len(claim_report.issues)} 条 claim 需要复核",
                    step=len(steps),
                )
                for issue in claim_report.issues[:3]:
                    warn(
                        f"语义复核：{issue.verdict} — {issue.claim[:80]}",
                        step=len(steps),
                    )

        duration_ms = (time.time() - started_wall) * 1000
        if answer:
            messages.append(Message("assistant", answer.strip()))
        if not final_emitted:
            emit(
                "final",
                step=len(steps),
                answer=answer,
                forced=forced_final,
                status=status,
            )
        result = AgentResult(
            task=task,
            answer=answer.strip(),
            status=status,
            steps=steps,
            citations=citations,
            usage=usage,
            trace_id=trace_id,
            duration_ms=duration_ms,
            warnings=warnings,
            forced_final=forced_final,
            metadata={
                "provider": self.llm.name,
                "model": getattr(self.config.llm, "model", ""),
                "tool_mode": self.config.llm.tool_mode,
                "kb": self.kb_description,
                "session_id": session_id,
                "plan": list(ctx.plan),
                "budget": budget.to_dict(),
                "evidence_profile": profile.to_dict(),
                "requested_model": ctx.requested_model,
                "evidence_count": evidence_count,
                "evidence_retries": evidence_retries,
                "grounding": grounding.to_dict() if grounding is not None else None,
                "claim_verification": (
                    claim_report.to_dict() if claim_report is not None else None
                ),
                "citation_alignment": (
                    citation_alignment.to_dict() if citation_alignment is not None else None
                ),
                "fact_coverage": (
                    fact_coverage_report.to_dict() if fact_coverage_report is not None else None
                ),
                "config_path": self.config.config_path,
                "general_fallback": general_fallback_mode,
                "trace_path": str(tracer.path) if tracer.path else "",
            },
            transcript=messages,
        )

        if self.memory is not None and persist_episode:
            try:
                episode_id = self.memory.save_episode(result, session_id=session_id)
                if episode_id:
                    result.metadata["episode_id"] = episode_id
            except Exception:
                logger.exception("保存长期记忆失败")

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
        return result

    # ------------------------------------------------------------------
    # LLM 调用与收尾
    # ------------------------------------------------------------------
    def _complete(
        self,
        messages: Sequence[Message],
        *,
        use_tools: bool,
        warnings: List[str],
    ) -> Optional[LLMResponse]:
        """带一次重试的 LLM 调用；失败返回 None。"""
        tools_schema = None
        if use_tools and self.config.llm.tool_mode == "native":
            tools_schema = self.tools.schema(self.enabled_tools)
        last_error: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                return self.llm.complete(
                    messages,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    tools=tools_schema,
                )
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    warnings.append(f"LLM 调用失败，1 秒后重试：{exc}")
                    time.sleep(max(0.0, float(self.config.agent.retry_delay)))
        logger.error("LLM 调用最终失败: %s", last_error)
        return None

    def _force_final(
        self,
        messages: List[Message],
        *,
        usage: Usage,
        warnings: List[str],
        emit: Callable[..., None],
        warn: Callable[..., None],
        status: str,
    ) -> Optional[str]:
        """在预算耗尽时要求模型基于已有 observation 收尾。"""
        if usage.llm_calls >= self.policy.max_llm_calls:
            return None
        reason_text = {
            "max_steps": "步数达到上限",
            "max_tool_calls": "工具调用次数达到上限",
            "max_llm_calls": "LLM 调用次数达到上限",
            "timeout": "运行超时",
            "no_answer": "没有产生最终答案",
        }.get(status, status)

        messages.append(Message("user", format_forced_final(reason_text)))
        resp = self._complete(messages, use_tools=False, warnings=warnings)
        if resp is None:
            return None
        usage.merge(resp.usage)
        usage.llm_calls += 1
        emit(
            "llm_response",
            step=len(messages),
            text=truncate_text(resp.text, 2000),
            tool_calls=[],
            forced=True,
        )
        parsed: ParsedAction = parse_action(resp.text, plain_text_final=True)
        if parsed.kind == "final" and parsed.final_answer.strip():
            return parsed.final_answer.strip()
        if parsed.kind == "tool":
            warn("预算耗尽后模型仍尝试调用工具，已忽略", step=len(messages))
        else:
            warn(f"收尾生成未得到有效答案：{parsed.error}", step=len(messages))
        return None

    @staticmethod
    def _fallback_answer(steps: Sequence[AgentStep], status: str) -> str:
        blocks = [
            f"- {step.tool_call.name if step.tool_call else 'tool'}：{step.result.content}"
            for step in steps
            if step.result is not None and step.result.ok and step.result.content
        ]
        reason = {
            "max_steps": "达到步数上限",
            "max_tool_calls": "达到工具调用上限",
            "max_llm_calls": "达到 LLM 调用上限",
            "timeout": "运行超时",
            "error": "运行出错",
        }.get(status, "未能完成推理")
        if not blocks:
            return (
                f"抱歉，{reason}，且没有从手册中检索到可用信息。"
                "请补充具体机型或问题细节后重试。"
            )
        return (
            f"抱歉，{reason}。以下是已从手册中检索到的原始信息，供参考：\n\n"
            + truncate_text("\n".join(blocks), 2400)
        )

    @staticmethod
    def _collect_evidence_texts(result: Any, store: Dict[str, str]) -> None:
        """把工具结果里的片段文本按引用编号缓存，供 grounding 使用。"""
        if result is None:
            return
        data = getattr(result, "data", {}) or {}
        by_chunk: Dict[str, str] = {}
        for passage in data.get("passages") or []:
            if not isinstance(passage, dict):
                continue
            chunk_id = str(passage.get("chunk_id") or "")
            text = str(passage.get("text") or "")
            if chunk_id and text:
                by_chunk[chunk_id] = text
        fallback_text = str(data.get("text") or "")
        for citation in getattr(result, "citations", []) or []:
            ref = str(getattr(citation, "ref", "") or "").upper()
            chunk_id = str(getattr(citation, "chunk_id", "") or "")
            if not ref:
                continue
            text = by_chunk.get(chunk_id) or fallback_text
            if text:
                store[ref] = text

    @staticmethod
    def _is_evidence_result(call: ToolCall, result: Any) -> bool:
        """判断一次工具调用是否真的带来了手册证据。"""
        if result is None or not getattr(result, "ok", False):
            return False
        if call.name == "manual_search":
            try:
                return int((result.data or {}).get("count") or 0) > 0
            except (TypeError, ValueError):
                return bool(result.content)
        if call.name == "manual_fetch":
            return bool(result.content)
        return False

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """返回组件健康状态，供 CLI / HTTP 探活。"""
        kb_stats: Any = None
        if self.kb is not None:
            try:
                kb_stats = self.kb.stats()
            except Exception as exc:
                kb_stats = {"error": str(exc)}
        return {
            "agent": self.config.agent.name,
            "llm": self.llm.health(),
            "tools": self.tools.describe(self.enabled_tools),
            "knowledge": kb_stats,
            "memory": self.memory.stats() if self.memory is not None else {"available": False},
            "policy": self.policy.to_dict(),
            "system_prompt_chars": len(self.system_prompt),
        }

    def close(self) -> None:
        try:
            self.tools.close()
        except Exception:
            pass


__all__ = ["AgentRuntime"]
