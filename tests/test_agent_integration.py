"""端到端集成：离线策略 + JSONL 语料 + 记忆 + 事件流。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent import Agent, AgentConfig
from agent.policy import AgentPolicy


def _offline_agent(
    tmp_path: Path,
    corpus: str,
    *,
    memory: bool = False,
    max_steps: int = 4,
) -> Agent:
    config = AgentConfig()
    config.llm.provider = "offline"
    config.llm.model = "offline-policy"
    config.knowledge.backend = "jsonl"
    config.knowledge.jsonl_path = corpus
    config.agent.max_steps = max_steps
    config.agent.max_seconds = 30
    config.agent.retry_delay = 0.0
    config.memory.enabled = memory
    config.memory.db_path = str(tmp_path / "memory.db")
    config.trace.enabled = False
    return Agent(config, policy=AgentPolicy.from_config(config))


def test_offline_end_to_end(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus)
    try:
        result = agent.ask("智能飞行电池在低温环境下如何保养？")
        assert result.status == "ok"
        assert result.steps[0].tool_call.name == "manual_search"
        assert "电池" in result.answer
        assert result.citations
        assert result.citations[0].ref.startswith("S")
        assert result.usage.tool_calls == 1
        assert not result.forced_final
    finally:
        agent.close()


def test_offline_retries_with_simplified_query(tmp_path):
    """第一次检索为空时，离线策略会换关键词再搜一次。"""
    corpus_path = tmp_path / "empty_first.jsonl"
    corpus_path.write_text(
        '{"chunk_id":"x1","text":"返航高度设置：起飞前请在 DJI Fly 中设置返航高度。","title":"手册","page":1}\n',
        encoding="utf-8",
    )
    agent = _offline_agent(tmp_path, str(corpus_path))
    try:
        result = agent.ask("请问返航高度应该怎么设置？")
        searches = [
            step for step in result.steps
            if step.tool_call is not None and step.tool_call.name == "manual_search"
        ]
        assert result.status == "ok"
        assert "返航" in result.answer
        assert searches
    finally:
        agent.close()


def test_stream_events(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus)
    try:
        events = list(agent.stream("固件升级失败怎么办？"))
        types = [event.type for event in events]
        assert "run_start" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "final" in types
        assert "run_end" in types
        assert "result" in types
        final_event = next(event for event in events if event.type == "final")
        assert final_event.payload["answer"]
        result_event = next(event for event in events if event.type == "result")
        assert result_event.payload["status"] == "ok"
        assert result_event.payload["citations"]
    finally:
        agent.close()


def test_session_memory_and_turns(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus, memory=True)
    try:
        first = agent.ask("智能飞行电池怎么保养？", session_id="session-1")
        second = agent.ask("固件升级失败怎么办？", session_id="session-1")
        assert first.status == "ok" and second.status == "ok"
        assert "电池" in first.answer
        assert "固件" in second.answer

        turns = agent.memory.get_turns("session-1")
        assert len(turns) == 4
        assert turns[0]["role"] == "user"
        assert turns[1]["role"] == "assistant"

        episodes = agent.memory.recent_episodes()
        assert len(episodes) == 2
    finally:
        agent.close()


def test_calculator_only_task(tmp_path):
    agent = _offline_agent(tmp_path, str(tmp_path / "no_corpus.jsonl"))
    try:
        result = agent.ask("请计算 (5000-249)/1000")
        assert result.status == "ok"
        assert result.steps[0].tool_call.name == "calculator"
        assert "4.751" in result.answer
    finally:
        agent.close()


def test_health_reports_components(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus)
    try:
        health = agent.health()
        assert health["llm"]["provider"] == "offline"
        assert health["knowledge"]["count"] == 4
        assert health["memory"]["available"] is False
        assert health["policy"]["max_steps"] == 4
    finally:
        agent.close()
