"""LangGraph 适配层测试。"""

from __future__ import annotations

from tests.helpers import action, final


def test_graph_engine_matches_legacy(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池"),
            final("电池需要预热。[S1]"),
        ],
        corpus_path=sample_corpus,
        engine="graph",
    )
    result = agent.ask("电池低温怎么保养？")
    assert result.status == "ok"
    assert result.answer == "电池需要预热。[S1]"
    assert agent.health()["engine"] == "langgraph"


def test_legacy_is_default(make_agent, sample_corpus):
    agent = make_agent([final("ok")], corpus_path=sample_corpus)
    assert "engine" not in agent.health()
