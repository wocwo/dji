"""配置加载、环境变量覆盖与校验。"""

from __future__ import annotations

import os

import pytest

from agent.config import AgentConfig, LLMSettings, load_env_file
from agent.errors import ConfigError


def test_env_file_loading(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENT_ENV_FILE", raising=False)
    (tmp_path / ".env.agent").write_text(
        "# local secret\nTEST_AGENT_ENV_KEY=hello\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_AGENT_ENV_KEY", raising=False)
    try:
        load_env_file()
        assert os.environ["TEST_AGENT_ENV_KEY"] == "hello"
    finally:
        os.environ.pop("TEST_AGENT_ENV_KEY", None)


def test_defaults():
    config = AgentConfig()
    assert config.agent.max_steps == 8
    assert config.llm.provider == "ollama"
    assert config.knowledge.backend == "auto"
    assert config.trace.enabled is True


def test_from_dict_partial():
    config = AgentConfig.from_dict(
        {"agent": {"max_steps": 3}, "llm": {"provider": "offline", "model": "x"}}
    )
    assert config.agent.max_steps == 3
    assert config.llm.provider == "offline"
    assert config.memory.db_path == ".agent/memory.db"


def test_from_yaml(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "agent:\n  max_steps: 4\nllm:\n  provider: offline\n  model: offline-policy\n",
        encoding="utf-8",
    )
    config = AgentConfig.from_yaml(str(path))
    assert config.agent.max_steps == 4
    assert config.llm.provider == "offline"
    assert config.config_path.endswith("agent.yaml")


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(ConfigError):
        AgentConfig.from_yaml(str(tmp_path / "nope.yaml"))


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "offline")
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    monkeypatch.setenv("AGENT_KB_BACKEND", "jsonl")
    monkeypatch.setenv("AGENT_TRACE_ENABLED", "false")
    config = AgentConfig.from_dict({})
    config.apply_env()
    assert config.llm.provider == "offline"
    assert config.agent.max_steps == 3
    assert config.knowledge.backend == "jsonl"
    assert config.trace.enabled is False


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_LLM_URL", "http://example.com:9999")
    path = tmp_path / "agent.yaml"
    path.write_text("llm:\n  base_url: ${MY_LLM_URL}\n", encoding="utf-8")
    config = AgentConfig.from_yaml(str(path))
    assert config.llm.base_url == "http://example.com:9999"


def test_invalid_tool_mode():
    with pytest.raises(ConfigError):
        LLMSettings(tool_mode="wrong")


def test_validate_rejects_unknown_provider():
    config = AgentConfig.from_dict({"llm": {"provider": "unknown"}})
    with pytest.raises(ConfigError):
        config.validate()


def test_validate_accepts_vllm_alias():
    config = AgentConfig.from_dict({"llm": {"provider": "vllm"}})
    config.validate()


def test_unknown_keys_are_ignored():
    config = AgentConfig.from_dict({"llm": {"provider": "offline", "surprise": 1}})
    assert config.llm.provider == "offline"
    assert getattr(config.llm, "_unknown_keys") == ["surprise"]


def test_to_dict_roundtrip():
    config = AgentConfig.from_dict({"agent": {"max_steps": 2}, "llm": {"provider": "offline"}})
    data = config.to_dict()
    assert data["agent"]["max_steps"] == 2
    assert data["llm"]["provider"] == "offline"


def test_deepseek_env_defaults(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_BASE_URL", raising=False)
    config = AgentConfig.from_dict({})
    config.apply_env()
    assert config.llm.provider == "deepseek"
    assert config.llm.base_url == "https://api.deepseek.com"
    assert config.llm.model == "deepseek-flash"
    assert config.llm.api_key == "sk-test-key"


def test_deepseek_config_file_loads(monkeypatch):
    monkeypatch.delenv("AGENT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("AGENT_LLM_BASE_URL", raising=False)
    config = AgentConfig.from_yaml("configs/agent.deepseek.yaml")
    assert config.llm.provider == "deepseek"
    assert config.llm.base_url == "https://api.deepseek.com"
    assert config.llm.model == "deepseek-flash"
    assert config.llm.tool_mode == "native"
    assert config.llm.thinking is True
    assert config.llm.reasoning_effort == "high"


def test_local_config_file_loads():
    config = AgentConfig.from_yaml("configs/agent.local.yaml")
    assert config.llm.provider == "ollama"
    assert config.llm.model == "qwen3.5:4b"
    assert config.llm.tool_mode == "native"
    assert config.llm.thinking is False
    assert config.llm.num_ctx == 16384


def test_project_yaml_loads():
    config = AgentConfig.from_yaml("configs/agent.yaml")
    assert config.agent.name == "dji-manual-agent"
    assert "manual_search" in config.tools.enabled
