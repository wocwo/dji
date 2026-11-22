"""LLM 客户端协议测试：DeepSeek OpenAI 兼容接口的 URL 与请求体。"""

from __future__ import annotations

from typing import Any, Dict

import pytest

import agent.llm as llm_module
from agent.config import LLMSettings
from agent.llm import OllamaClient, OpenAIClient
from agent.types import Message


def test_deepseek_endpoint_has_no_v1_prefix():
    client = OpenAIClient(
        LLMSettings(provider="deepseek", base_url="https://api.deepseek.com")
    )
    assert client._endpoint("/chat/completions") == "https://api.deepseek.com/chat/completions"
    assert client._endpoint("/models") == "https://api.deepseek.com/models"


def test_deepseek_endpoint_accepts_explicit_v1():
    client = OpenAIClient(
        LLMSettings(provider="deepseek", base_url="https://api.deepseek.com/v1")
    )
    assert client._endpoint("/chat/completions") == "https://api.deepseek.com/v1/chat/completions"


def test_deepseek_payload_sends_thinking(monkeypatch):
    captured: Dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> Dict[str, Any]:
            return {
                "choices": [{"message": {"content": "你好"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "model": "deepseek-flash",
            }

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        captured["headers"] = kwargs.get("headers")
        return FakeResponse()

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    settings = LLMSettings(
        provider="deepseek",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        thinking=True,
        reasoning_effort="high",
        tool_mode="native",
    )
    client = OpenAIClient(settings)
    response = client.complete([Message("user", "你好")])
    assert response.text == "你好"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "deepseek-flash"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_ollama_payload_sends_num_ctx_and_think(monkeypatch):
    captured: Dict[str, Any] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> Dict[str, Any]:
            return {
                "message": {"content": "你好"},
                "prompt_eval_count": 12,
                "eval_count": 3,
                "model": "qwen3.5:4b",
                "done_reason": "stop",
            }

    def fake_post(url: str, **kwargs: Any) -> FakeResponse:
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(llm_module.requests, "post", fake_post)
    settings = LLMSettings(
        provider="ollama",
        model="qwen3.5:4b",
        base_url="http://localhost:11434",
        thinking=False,
        num_ctx=16384,
        tool_mode="native",
    )
    client = OllamaClient(settings)
    response = client.complete([Message("user", "你好")])
    assert response.text == "你好"
    payload = captured["payload"]
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 16384
    assert captured["url"].endswith("/api/chat")
