"""CLI 测试：config / tools / demo / run。"""

from __future__ import annotations

import json

from agent.cli import main


def _json_from(text: str, opener: str = "{") -> dict:
    start = text.index(opener)
    return json.loads(text[start:])


def test_config_command(capsys):
    code = main(["config", "--provider", "offline", "--no-trace", "--no-memory"])
    assert code == 0
    out = capsys.readouterr().out
    data = _json_from(out, "{")
    assert data["llm"]["provider"] == "offline"


def test_tools_command(capsys, sample_corpus):
    code = main(
        [
            "tools",
            "--provider", "offline",
            "--kb", "jsonl",
            "--jsonl", sample_corpus,
            "--no-memory",
            "--no-trace",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    tools = _json_from(out, "[")
    names = {tool["name"] for tool in tools}
    assert "manual_search" in names
    assert "calculator" in names


def test_demo_command(capsys, sample_corpus):
    code = main(
        [
            "demo",
            "--kb", "jsonl",
            "--jsonl", sample_corpus,
            "--no-memory",
            "--no-trace",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    data = _json_from(out, "{")
    assert data["status"] == "ok"
    assert "电池" in data["answer"]
    assert data["citations"]


def test_run_command(capsys):
    code = main(
        [
            "run", "计算", "1+1",
            "--provider", "offline",
            "--kb", "none",
            "--no-memory",
            "--no-trace",
            "--json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    data = _json_from(out, "{")
    assert data["status"] == "ok"
    assert "2" in data["answer"]
    assert data["steps"][0]["tool_call"]["name"] == "calculator"


def test_run_empty_task_rejected():
    code = main(["run", "   ", "--provider", "offline", "--kb", "none", "--no-memory", "--no-trace"])
    assert code == 2
