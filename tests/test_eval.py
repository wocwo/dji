"""评测框架测试。"""

from __future__ import annotations

from pathlib import Path

from agent import Agent, AgentConfig
from agent.eval import EvalCase, format_report, run_eval
from agent.policy import AgentPolicy


def _offline_agent(tmp_path: Path, corpus: str) -> Agent:
    config = AgentConfig()
    config.llm.provider = "offline"
    config.knowledge.backend = "jsonl"
    config.knowledge.jsonl_path = corpus
    config.agent.max_steps = 4
    config.agent.retry_delay = 0.0
    config.memory.enabled = False
    config.trace.enabled = False
    return Agent(config, policy=AgentPolicy.from_config(config))


def test_run_eval_all_pass(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus)
    try:
        cases = [
            EvalCase(
                id="battery",
                task="智能飞行电池在低温环境下如何保养？",
                expected_tools=["manual_search"],
                answer_contains=["电池"],
                min_citations=1,
            ),
            EvalCase(
                id="math",
                task="请计算 (5000-249)/1000",
                expected_tools=["calculator"],
                answer_contains=["4.751"],
            ),
        ]
        report = run_eval(agent, cases, verbose=False)
        assert report["total"] == 2
        assert report["passed"] == 2
        assert report["success_rate"] == 1.0
        assert report["tool_accuracy"] == 1.0
        assert report["answer_accuracy"] == 1.0
        assert report["citation_accuracy"] == 1.0
        assert format_report(report)
    finally:
        agent.close()


def test_run_eval_reports_failure(tmp_path, sample_corpus):
    agent = _offline_agent(tmp_path, sample_corpus)
    try:
        cases = [
            EvalCase(
                id="wrong-tool",
                task="智能飞行电池怎么保养？",
                expected_tools=["calculator"],
                answer_contains=["完全不存在的关键词"],
            )
        ]
        report = run_eval(agent, cases, verbose=False)
        assert report["passed"] == 0
        assert report["cases"][0]["checks"]["tools"] is False
        assert report["cases"][0]["checks"]["answer"] is False
    finally:
        agent.close()
