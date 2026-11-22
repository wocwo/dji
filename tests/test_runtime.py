"""运行时循环测试：工具调用、恢复、预算、重复拦截、原生调用与追踪。"""

from __future__ import annotations

import json

from pathlib import Path

from agent.llm import LLMResponse
from agent.trace import read_trace
from agent.types import ToolCall
from tests.helpers import action, final


def test_tool_then_final(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池 低温 保养"),
            final("低温下不要立即充电，应先预热电池。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("智能飞行电池低温环境下如何保养？")
    assert result.status == "ok"
    assert len(result.steps) == 2
    assert result.steps[0].tool_call.name == "manual_search"
    assert result.steps[0].result.ok is True
    assert result.citations
    assert result.citations[0].ref == "S1"
    assert result.usage.tool_calls == 1
    assert result.usage.llm_calls == 2
    assert "低温" in result.answer


def test_invalid_action_recovery(make_agent, sample_corpus):
    agent = make_agent(
        ["这不是一个 JSON 动作", final("我改用直接回答。")],
        corpus_path=sample_corpus,
        plain_text_final=False,
    )
    result = agent.ask("测试")
    assert result.status == "ok"
    assert result.steps[0].error
    assert any("解析失败" in warning for warning in result.warnings)
    assert result.answer == "我改用直接回答。"


def test_max_steps_forces_final(make_agent):
    agent = make_agent(
        [action("calculator", expression="1+1"), final("计算结果是 2。")],
        enabled_tools=["calculator"],
        max_steps=1,
    )
    result = agent.ask("1+1 等于几？")
    assert result.status == "max_steps"
    assert result.forced_final is True
    assert result.answer == "计算结果是 2。"
    assert result.usage.tool_calls == 1


def test_duplicate_tool_call_blocked(make_agent, sample_corpus):
    same = action("manual_search", query="电池")
    agent = make_agent(
        [same, same, final("电池保养要点见 [S1]。")],
        corpus_path=sample_corpus,
    )
    result = agent.ask("电池怎么保养？")
    assert result.status == "ok"
    assert result.usage.tool_calls == 1
    assert any("重复" in warning for warning in result.warnings)
    assert any(step.error == "重复调用" for step in result.steps)


def test_tool_error_does_not_crash(make_agent):
    agent = make_agent(
        [action("no_such_tool", x=1), final("我换一种方式回答。")],
        enabled_tools=["calculator"],
    )
    result = agent.ask("测试未知工具")
    assert result.status == "ok"
    assert result.steps[0].result.ok is False
    assert "工具不存在" in result.steps[0].result.content


def test_policy_denies_tool(make_agent):
    agent = make_agent(
        [action("calculator", expression="1+1"), final("无法调用计算器。")],
        enabled_tools=["calculator"],
        allowed_tools=[],
    )
    result = agent.ask("1+1")
    assert result.status == "ok"
    assert result.steps[0].result.ok is False
    assert "不在允许列表" in result.steps[0].result.content


def test_native_tool_calling(make_agent):
    native_call = LLMResponse(
        text="",
        tool_calls=[ToolCall(name="calculator", arguments={"expression": "2+2"})],
    )
    agent = make_agent(
        [native_call, final("结果是 4。")],
        enabled_tools=["calculator"],
        tool_mode="native",
    )
    result = agent.ask("2+2 等于几？")
    assert result.status == "ok"
    assert result.steps[0].tool_call.name == "calculator"
    assert result.steps[0].result.ok is True
    assert any(message.role == "tool" for message in result.transcript)
    tool_message = next(message for message in result.transcript if message.role == "tool")
    assert tool_message.tool_call_id == native_call.tool_calls[0].id


def test_llm_failure_returns_fallback(make_agent):
    agent = make_agent([], enabled_tools=[])
    result = agent.ask("无论什么问题")
    assert result.status == "error"
    assert result.answer
    assert any("LLM" in warning for warning in result.warnings)


def test_no_citation_warning(make_agent, sample_corpus):
    agent = make_agent(
        [action("manual_search", query="电池 低温"), final("电池应避免低温充电。")],
        corpus_path=sample_corpus,
    )
    result = agent.ask("电池低温怎么办？")
    assert result.citations
    assert any("未标注引用" in warning for warning in result.warnings)


def test_trace_written(make_agent, sample_corpus):
    agent = make_agent(
        [action("manual_search", query="电池"), final("见 [S1]。")],
        corpus_path=sample_corpus,
        trace=True,
    )
    result = agent.ask("电池")
    trace_path = result.metadata.get("trace_path")
    assert trace_path and Path(trace_path).exists()
    events = read_trace(trace_path)
    types = [event["type"] for event in events]
    assert "run_start" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert "run_end" in types


def test_native_plain_text_final(make_agent):
    """native 模式下模型直接返回自然语言，即使关闭纯文本兜底也应被接受。"""
    agent = make_agent(
        [LLMResponse(text="这是直接回答。")],
        enabled_tools=["calculator"],
        tool_mode="native",
        plain_text_final=False,
    )
    result = agent.ask("测试")
    assert result.status == "ok"
    assert result.answer == "这是直接回答。"


def test_forced_final_emits_final_event(make_agent):
    from agent.types import AgentEvent

    events = []
    agent = make_agent(
        [action("calculator", expression="1+1"), final("2")],
        enabled_tools=["calculator"],
        max_steps=1,
    )
    result = agent.ask("1+1", on_event=events.append)
    assert result.status == "max_steps"
    finals = [event for event in events if event.type == "final"]
    assert finals
    assert finals[0].payload.get("forced") is True


def test_evidence_gate_auto_searches(make_agent, sample_corpus):
    """模型直接给手册答案但没有证据时，运行时自动检索并让它重答。"""
    agent = make_agent(
        [
            final("低温下电池要先预热。"),
            final("低温下电池要先预热至 10℃ 以上。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("电池低温怎么保养？")
    assert result.status == "ok"
    searched = [
        step for step in result.steps
        if step.tool_call is not None and step.tool_call.name == "manual_search"
    ]
    assert searched, "gate should auto-run manual_search"
    assert result.citations
    assert any("RAG 回退" in warning for warning in result.warnings)
    assert result.metadata["evidence_profile"]["requires_evidence"] is True
    assert result.metadata["evidence_count"] == 1


def test_evidence_gate_allows_general_answer(make_agent):
    agent = make_agent([final("你好，有什么可以帮你？")], enabled_tools=["calculator"])
    result = agent.ask("你好")
    assert result.status == "ok"
    assert not any(step.tool_call for step in result.steps)
    assert result.metadata["evidence_profile"]["requires_evidence"] is False
    assert result.metadata["grounding"] is None


def test_evidence_gate_can_be_disabled(make_agent, sample_corpus):
    agent = make_agent(
        [final("电池要定期保养。")],
        corpus_path=sample_corpus,
        evidence_guard=False,
    )
    result = agent.ask("电池怎么保养？")
    assert result.status == "ok"
    assert not any(
        step.tool_call is not None and step.tool_call.name == "manual_search"
        for step in result.steps
    )


def test_evidence_gate_rejects_draft_step(make_agent, sample_corpus):
    agent = make_agent(
        [final("无证据的草稿"), final("有证据的答案。[S1]")],
        corpus_path=sample_corpus,
    )
    result = agent.ask("固件升级失败怎么办？")
    rejected = [step for step in result.steps if step.error == "缺少手册证据"]
    assert rejected


def test_evidence_gate_does_not_repeat_empty_search(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="量子计算机"),
            final("手册里没有提到。"),
            final("根据当前手册内容，未找到相关信息。"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("量子计算机怎么保养？")
    searches = [
        step for step in result.steps
        if step.tool_call is not None and step.tool_call.name == "manual_search"
    ]
    assert len(searches) == 1, "already tried an empty search; should not auto-repeat"
    assert result.status == "ok"
    assert any("未获得手册证据" in warning for warning in result.warnings)


def test_grounding_flags_numeric_mismatch(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池 低温"),
            final("建议将电池预热到 99℃。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("电池低温怎么保养？")
    grounding = result.metadata["grounding"]
    assert grounding["passed"] is False
    assert any(issue["kind"] == "numeric_mismatch" for issue in grounding["issues"])
    assert any("引用校验" in warning for warning in result.warnings)
    assert any("数值" in warning for warning in result.warnings)


def test_runtime_normalizes_bare_citations(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池"),
            final("见 S1S1"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("电池怎么保养？")
    assert result.answer == "见 [S1][S1]"
    assert [citation.ref for citation in result.citations] == ["S1"]


def test_requested_model_scopes_search(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池"),
            final("电池需要预热。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    result = agent.ask("Mavic 4 Pro 电池低温怎么保养？")
    search_steps = [
        step for step in result.steps
        if step.tool_call is not None and step.tool_call.name == "manual_search"
    ]
    assert search_steps
    data = search_steps[0].result.data
    assert data["requested_model"] == "Mavic 4 Pro"
    assert all("Mavic 4 Pro" in passage["model"] for passage in data["passages"])
    assert result.metadata["requested_model"] == "Mavic 4 Pro"


def test_claim_verifier_runtime(make_agent, sample_corpus):
    verdicts = [
        {
            "claim": "电池应该冷藏。",
            "verdict": "unsupported",
            "reason": "手册要求预热而不是冷藏",
        }
    ]
    agent = make_agent(
        [
            action("manual_search", query="电池"),
            final("电池应该冷藏。[S1]"),
            json.dumps(verdicts, ensure_ascii=False),
        ],
        corpus_path=sample_corpus,
        claim_verifier=True,
    )
    result = agent.ask("电池低温怎么保养？")
    report = result.metadata["claim_verification"]
    assert report["checked"] == 1
    assert report["issues"][0]["verdict"] == "unsupported"
    assert any("语义复核" in warning for warning in result.warnings)
    assert result.usage.llm_calls >= 3


def test_fact_coverage_retries_once(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池"),
            final("电池要满电。"),
            final("电池要满电、-10℃ 以下不能飞、先预热，5℃ 以下不要充电。[S1]"),
        ],
        corpus_path=sample_corpus,
        fact_coverage=True,
        max_steps=5,
    )
    result = agent.ask("Mavic 4 Pro 电池低温怎么保养？")
    assert result.status == "ok"
    assert any(step.error == "关键事实未覆盖" for step in result.steps)
    assert "-10℃" in result.answer
    assert result.metadata["fact_coverage"]["passed"] is True


def test_system_prompt_protocol_is_valid_json_example(make_agent):
    agent = make_agent([final("ok")], enabled_tools=["calculator"])
    prompt = agent.system_prompt
    assert '{"thought"' in prompt
    assert '{{"thought"' not in prompt
    assert '"action_input": {"参数名"' in prompt


def test_health_shape(make_agent, sample_corpus):
    agent = make_agent([final("ok")], corpus_path=sample_corpus)
    health = agent.health()
    assert health["llm"]["healthy"] is True
    assert health["agent"] == "dji-manual-agent"
    assert health["knowledge"]["count"] == 4
    assert any(tool["name"] == "manual_search" for tool in health["tools"])
