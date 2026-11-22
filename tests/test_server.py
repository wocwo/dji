"""HTTP API 测试。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agent.server import create_app  # noqa: E402
from tests.helpers import action, final  # noqa: E402


def test_tools_and_run(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="电池 低温"),
            final("低温下不要立即充电。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    app = create_app(agent=agent)
    with TestClient(app) as client:
        tools = client.get("/v1/tools")
        assert tools.status_code == 200
        assert any(item["name"] == "manual_search" for item in tools.json())

        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["knowledge"]["count"] == 4

        config = client.get("/v1/config")
        assert config.status_code == 200
        assert config.json()["llm"]["api_key"] in (None, "***")

        response = client.post("/v1/agent/run", json={"task": "电池低温怎么保养？"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "低温" in data["answer"]
        assert data["citations"]


def test_stream_endpoint(make_agent, sample_corpus):
    agent = make_agent(
        [
            action("manual_search", query="固件升级"),
            final("请重新下载固件。[S1]"),
        ],
        corpus_path=sample_corpus,
    )
    app = create_app(agent=agent)
    with TestClient(app) as client:
        response = client.post("/v1/agent/stream", json={"task": "固件升级失败怎么办？"})
        assert response.status_code == 200
        assert "event: tool_call" in response.text
        assert "event: final" in response.text
        assert "event: result" in response.text
        # data 应为业务 payload，而不是再包一层 {type,step,payload}
        assert '"name": "manual_search"' in response.text
        assert '"type": "tool_call"' not in response.text


def test_web_ui_is_served(make_agent, sample_corpus):
    agent = make_agent([final("ok")], corpus_path=sample_corpus)
    app = create_app(agent=agent)
    with TestClient(app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert "DJI 手册 Agent" in index.text
        assert "/static/app.js" in index.text

        script = client.get("/static/app.js")
        assert script.status_code == 200
        assert "renderMarkdown" in script.text

        styles = client.get("/static/styles.css")
        assert styles.status_code == 200
        assert "--teal" in styles.text


def test_run_accepts_history(make_agent, sample_corpus):
    agent = make_agent([final("历史已收到")], corpus_path=sample_corpus)
    app = create_app(agent=agent)
    with TestClient(app) as client:
        response = client.post(
            "/v1/agent/run",
            json={
                "task": "继续",
                "session_id": "test-session",
                "history": [
                    {"role": "user", "content": "上一轮问题"},
                    {"role": "assistant", "content": "上一轮回答"},
                ],
            },
        )
        assert response.status_code == 200
        assert response.json()["answer"] == "历史已收到"


def test_run_endpoint_validation(make_agent, sample_corpus):
    agent = make_agent([final("ok")], corpus_path=sample_corpus)
    app = create_app(agent=agent)
    with TestClient(app) as client:
        response = client.post("/v1/agent/run", json={"task": ""})
        assert response.status_code == 422
