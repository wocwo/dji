"""多用户隔离测试。"""

from __future__ import annotations

import base64
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agent.server import create_app  # noqa: E402
from tests.helpers import final  # noqa: E402


def _auth(user: str, password: str) -> dict:
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def _wait(client: TestClient, job_id: str, headers: dict) -> dict:
    for _ in range(80):
        response = client.get(f"/v1/agent/jobs/{job_id}", headers=headers)
        if response.status_code == 200 and response.json().get("status") in ("done", "error"):
            return response.json()
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_users_cannot_read_each_others_jobs_or_sessions(make_agent, sample_corpus, monkeypatch):
    monkeypatch.setenv("AGENT_USERS", '{"alice":"pw-a","bob":"pw-b"}')
    monkeypatch.delenv("AGENT_ACCESS_TOKEN", raising=False)
    agent = make_agent([final("alice answer [S1]"), final("bob answer [S1]")], corpus_path=sample_corpus, memory=True)
    app = create_app(agent=agent)
    with TestClient(app) as client:
        alice = _auth("alice", "pw-a")
        bob = _auth("bob", "pw-b")
        assert client.get("/v1/sessions").status_code == 401
        # URL query token 不能再作为认证方式
        assert client.get("/v1/sessions?access_token=pw-a").status_code == 401

        alice_job = client.post("/v1/agent/jobs", json={"task": "alice question", "session_id": "same"}, headers=alice)
        bob_job = client.post("/v1/agent/jobs", json={"task": "bob question", "session_id": "same"}, headers=bob)
        assert alice_job.status_code == 200
        assert bob_job.status_code == 200

        alice_result = _wait(client, alice_job.json()["job_id"], alice)
        bob_result = _wait(client, bob_job.json()["job_id"], bob)
        assert alice_result["result"]["answer"].startswith("alice")
        assert bob_result["result"]["answer"].startswith("bob")

        # job ID 不能跨用户读取
        assert client.get(f"/v1/agent/jobs/{alice_job.json()['job_id']}", headers=bob).status_code == 404
        assert client.get(f"/v1/agent/jobs/{bob_job.json()['job_id']}", headers=alice).status_code == 404

        alice_sessions = client.get("/v1/sessions", headers=alice).json()["sessions"]
        bob_sessions = client.get("/v1/sessions", headers=bob).json()["sessions"]
        assert [row["session_id"] for row in alice_sessions] == ["same"]
        assert [row["session_id"] for row in bob_sessions] == ["same"]

        alice_turns = client.get("/v1/sessions/same", headers=alice).json()["turns"]
        bob_turns = client.get("/v1/sessions/same", headers=bob).json()["turns"]
        assert any("alice question" in row["content"] for row in alice_turns)
        assert all("alice question" not in row["content"] for row in bob_turns)
        assert any("bob question" in row["content"] for row in bob_turns)
        assert all("bob question" not in row["content"] for row in alice_turns)
