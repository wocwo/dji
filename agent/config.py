"""Agent 配置：YAML + 环境变量覆盖。

配置与原有 RAG 的 ``config_center.py`` 解耦：RAG 配置面向“检索/生成参数”，
这里面向“Agent 运行策略”（步数、工具、记忆、追踪、LLM 后端）。

优先级：代码默认值 < YAML 文件 < 环境变量。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ConfigError

try:  # PyYAML 是项目已有依赖，但保持可降级
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


DEFAULT_CONFIG_PATH = "configs/agent.yaml"
DEFAULT_ENV_FILE = ".env.agent"
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_env_file(path: Optional[str] = None) -> None:
    """加载本地 ``.env.agent``（不覆盖已存在的环境变量）。

    支持 ``KEY=VALUE``、``export KEY=VALUE``、``#`` 注释和简单引号。
    API Key 建议放在这个文件里，而不是提交到仓库的 YAML。
    """
    env_path = Path(path or os.environ.get("AGENT_ENV_FILE") or DEFAULT_ENV_FILE).expanduser()
    if not env_path.exists():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        try:
            os.environ.setdefault(key, value)
        except (TypeError, ValueError):
            continue


# ---------------------------------------------------------------------------
# 分组配置
# ---------------------------------------------------------------------------

@dataclass
class LLMSettings:
    """LLM 后端设置。"""

    provider: str = "ollama"          # ollama | openai | deepseek | vllm | offline | scripted
    model: str = "deepseek-r1:7b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout: float = 120.0
    tool_mode: str = "text"           # text（通用 JSON 协议）| native（原生 function calling）
    # DeepSeek 思考模式（可选）：thinking=true/false，reasoning_effort=none/low/high/max
    thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    # 本地 Ollama 上下文窗口（Qwen3.5 支持 256K，但 4GB 显存建议 8K-16K）
    num_ctx: Optional[int] = None
    stop: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.provider = str(self.provider).lower().strip()
        self.tool_mode = str(self.tool_mode).lower().strip()
        if self.tool_mode not in ("text", "native"):
            raise ConfigError(f"llm.tool_mode 只能是 text/native，当前为 {self.tool_mode!r}")
        if isinstance(self.thinking, str):
            self.thinking = self.thinking.strip().lower() not in ("false", "0", "disabled", "no", "off")
        if self.reasoning_effort:
            self.reasoning_effort = str(self.reasoning_effort).lower().strip()


@dataclass
class KnowledgeSettings:
    """知识库检索设置。"""

    backend: str = "auto"             # auto | chroma | jsonl | none
    chroma_path: str = "./chroma_final"
    collection: str = "uav_manual_db"
    jsonl_path: str = "data/agent_demo_corpus.jsonl"
    top_k: int = 4
    candidate_limit: int = 300
    max_terms: int = 5
    search_timeout: float = 25.0
    max_passage_chars: int = 700

    def __post_init__(self) -> None:
        self.backend = str(self.backend).lower().strip()
        if self.backend not in ("auto", "chroma", "jsonl", "none"):
            raise ConfigError(
                f"knowledge.backend 只能是 auto/chroma/jsonl/none，当前为 {self.backend!r}"
            )


@dataclass
class MemorySettings:
    """长期记忆设置。"""

    enabled: bool = True
    db_path: str = ".agent/memory.db"
    history_turns: int = 6


@dataclass
class TraceSettings:
    """可观测性设置。"""

    enabled: bool = True
    dir: str = ".agent/traces"
    include_llm_output: bool = True


@dataclass
class ToolSettings:
    """工具设置。"""

    enabled: List[str] = field(
        default_factory=lambda: [
            "manual_search",
            "manual_fetch",
            "calculator",
            "clock",
            "update_plan",
            "save_note",
            "recall_notes",
        ]
    )
    allow_risky: bool = False
    timeout: float = 45.0

    def __post_init__(self) -> None:
        if isinstance(self.enabled, str):
            self.enabled = [x.strip() for x in self.enabled.split(",") if x.strip()]


@dataclass
class AgentSettings:
    """Agent 运行循环设置。"""

    name: str = "dji-manual-agent"
    max_steps: int = 8
    max_tool_calls: int = 12
    max_seconds: float = 180.0
    max_llm_calls: int = 10
    max_consecutive_errors: int = 3
    max_observation_chars: int = 4000
    max_history_chars: int = 8000
    plain_text_final: bool = True
    retry_delay: float = 1.0
    # 运行时引擎：legacy = 自研 ReAct；graph = LangGraph 适配层（实验）
    engine: str = "legacy"
    # 证据闸门：手册类问题如果模型直接作答、没有任何检索证据，自动触发 RAG 回退
    evidence_guard: bool = True
    evidence_auto_search: bool = True
    prefetch_search: bool = False
    max_evidence_retries: int = 1
    grounding_check: bool = True
    # 可选的第二模型复核：逐句判断引用是否真的支持 claim（默认关闭，DeepSeek 配置开启）
    claim_verifier: bool = False
    claim_verifier_max_claims: int = 12
    claim_verifier_high_risk_only: bool = False
    citation_align: bool = False
    fact_coverage: bool = False
    max_fact_retries: int = 1


# ---------------------------------------------------------------------------
# 总配置
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent 总配置。"""

    agent: AgentSettings = field(default_factory=AgentSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    knowledge: KnowledgeSettings = field(default_factory=KnowledgeSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    trace: TraceSettings = field(default_factory=TraceSettings)
    tools: ToolSettings = field(default_factory=ToolSettings)
    config_path: str = ""

    # -- 构造 -------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AgentConfig":
        data = data or {}
        try:
            return cls(
                agent=_build(AgentSettings, data.get("agent")),
                llm=_build(LLMSettings, data.get("llm")),
                knowledge=_build(KnowledgeSettings, data.get("knowledge")),
                memory=_build(MemorySettings, data.get("memory")),
                trace=_build(TraceSettings, data.get("trace")),
                tools=_build(ToolSettings, data.get("tools")),
            )
        except TypeError as exc:
            raise ConfigError(f"Agent 配置字段错误: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: Optional[str] = None) -> "AgentConfig":
        """从 YAML 加载；文件不存在时使用默认值而不是报错。"""
        load_env_file()
        resolved: Optional[Path] = None
        if path:
            resolved = Path(path).expanduser()
            if not resolved.exists():
                raise ConfigError(f"配置文件不存在: {resolved}")
        else:
            env_path = os.environ.get("AGENT_CONFIG")
            candidate = Path(env_path).expanduser() if env_path else Path(DEFAULT_CONFIG_PATH)
            if candidate.exists():
                resolved = candidate

        data: Dict[str, Any] = {}
        if resolved is not None:
            if yaml is None:
                raise ConfigError("需要 PyYAML 才能加载 YAML 配置: pip install pyyaml")
            try:
                raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                raise ConfigError(f"配置文件解析失败 {resolved}: {exc}") from exc
            if not isinstance(raw, dict):
                raise ConfigError(f"配置文件根节点必须是映射: {resolved}")
            data = _expand_env(raw)

        config = cls.from_dict(data)
        config.config_path = str(resolved) if resolved else ""
        config.apply_env()
        config.validate()
        return config

    # -- 覆盖与校验 -------------------------------------------------------
    def apply_env(self) -> "AgentConfig":
        """应用 ``AGENT_*`` 环境变量覆盖。"""
        mapping = [
            ("AGENT_LLM_PROVIDER", self.llm, "provider", str),
            ("AGENT_LLM_MODEL", self.llm, "model", str),
            ("AGENT_LLM_BASE_URL", self.llm, "base_url", str),
            ("AGENT_LLM_API_KEY", self.llm, "api_key", str),
            ("AGENT_TOOL_MODE", self.llm, "tool_mode", str),
            ("AGENT_LLM_THINKING", self.llm, "thinking", _to_bool),
            ("AGENT_LLM_REASONING_EFFORT", self.llm, "reasoning_effort", str),
            ("AGENT_LLM_NUM_CTX", self.llm, "num_ctx", int),
            ("AGENT_KB_BACKEND", self.knowledge, "backend", str),
            ("AGENT_CHROMA_PATH", self.knowledge, "chroma_path", str),
            ("AGENT_COLLECTION", self.knowledge, "collection", str),
            ("AGENT_JSONL_PATH", self.knowledge, "jsonl_path", str),
            ("AGENT_MEMORY_DB", self.memory, "db_path", str),
            ("AGENT_MEMORY_ENABLED", self.memory, "enabled", _to_bool),
            ("AGENT_TRACE_DIR", self.trace, "dir", str),
            ("AGENT_TRACE_ENABLED", self.trace, "enabled", _to_bool),
            ("AGENT_MAX_STEPS", self.agent, "max_steps", int),
            ("AGENT_ENGINE", self.agent, "engine", str),
            ("AGENT_MAX_TOOL_CALLS", self.agent, "max_tool_calls", int),
            ("AGENT_MAX_SECONDS", self.agent, "max_seconds", float),
            ("AGENT_EVIDENCE_GUARD", self.agent, "evidence_guard", _to_bool),
            ("AGENT_EVIDENCE_AUTO_SEARCH", self.agent, "evidence_auto_search", _to_bool),
            ("AGENT_MAX_EVIDENCE_RETRIES", self.agent, "max_evidence_retries", int),
            ("AGENT_GROUNDING_CHECK", self.agent, "grounding_check", _to_bool),
            ("AGENT_CLAIM_VERIFIER", self.agent, "claim_verifier", _to_bool),
            ("AGENT_CLAIM_VERIFIER_MAX_CLAIMS", self.agent, "claim_verifier_max_claims", int),
            ("AGENT_CLAIM_VERIFIER_HIGH_RISK_ONLY", self.agent, "claim_verifier_high_risk_only", _to_bool),
            ("AGENT_CITATION_ALIGN", self.agent, "citation_align", _to_bool),
            ("AGENT_FACT_COVERAGE", self.agent, "fact_coverage", _to_bool),
            ("AGENT_MAX_FACT_RETRIES", self.agent, "max_fact_retries", int),
            ("AGENT_ALLOW_RISKY", self.tools, "allow_risky", _to_bool),
            ("AGENT_TOOLS", self.tools, "enabled", lambda v: [x.strip() for x in v.split(",") if x.strip()]),
        ]
        for env_name, target, attr, caster in mapping:
            raw = os.environ.get(env_name)
            if raw is None or raw == "":
                continue
            try:
                setattr(target, attr, caster(raw))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"环境变量 {env_name}={raw!r} 无效: {exc}") from exc
        # API key 兜底：DeepSeek 用 DEEPSEEK_API_KEY，OpenAI 用 OPENAI_API_KEY
        if not self.llm.api_key:
            base_url = (self.llm.base_url or "").lower()
            if self.llm.provider == "deepseek" or "deepseek.com" in base_url:
                self.llm.api_key = os.environ.get("DEEPSEEK_API_KEY")
            elif self.llm.provider == "openai":
                self.llm.api_key = os.environ.get("OPENAI_API_KEY")
        # DeepSeek 官方 API 的默认地址与当前模型名
        if self.llm.provider == "deepseek":
            if not os.environ.get("AGENT_LLM_BASE_URL") and self.llm.base_url in (
                "",
                "http://localhost:11434",
            ):
                self.llm.base_url = "https://api.deepseek.com"
            if not os.environ.get("AGENT_LLM_MODEL") and self.llm.model in (
                "",
                "deepseek-r1:7b",
            ):
                self.llm.model = "deepseek-flash"
        # 环境变量切到离线策略时，模型名同步为可读的占位名
        if self.llm.provider == "offline" and self.llm.model == "deepseek-r1:7b":
            self.llm.model = "offline-policy"
        return self

    def validate(self) -> None:
        if self.agent.max_steps < 1:
            raise ConfigError("agent.max_steps 必须 >= 1")
        if self.agent.max_tool_calls < 0:
            raise ConfigError("agent.max_tool_calls 必须 >= 0")
        if self.agent.max_seconds <= 0:
            raise ConfigError("agent.max_seconds 必须 > 0")
        if self.agent.max_evidence_retries < 0:
            raise ConfigError("agent.max_evidence_retries 必须 >= 0")
        if self.agent.engine not in ("legacy", "graph"):
            raise ConfigError("agent.engine 必须是 legacy 或 graph")
        if self.agent.claim_verifier_max_claims <= 0:
            raise ConfigError("agent.claim_verifier_max_claims 必须 > 0")
        if self.agent.max_fact_retries < 0:
            raise ConfigError("agent.max_fact_retries 必须 >= 0")
        if self.llm.provider not in (
            "ollama",
            "openai",
            "vllm",
            "deepseek",
            "openai_compatible",
            "offline",
            "scripted",
        ):
            raise ConfigError(
                "llm.provider 仅支持 ollama/openai/vllm/deepseek/offline/scripted，"
                f"当前为 {self.llm.provider!r}"
            )
        if self.knowledge.top_k < 1:
            raise ConfigError("knowledge.top_k 必须 >= 1")

    # -- 导出 -------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        def dump(obj: Any) -> Dict[str, Any]:
            return {f.name: getattr(obj, f.name) for f in fields(obj)}

        return {
            "agent": dump(self.agent),
            "llm": dump(self.llm),
            "knowledge": dump(self.knowledge),
            "memory": dump(self.memory),
            "trace": dump(self.trace),
            "tools": dump(self.tools),
        }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _build(cls_: type, data: Optional[Dict[str, Any]]):
    """只取已知字段构造 dataclass，忽略 YAML 中的扩展键。"""
    if not data:
        return cls_()
    if not isinstance(data, dict):
        raise ConfigError(f"{cls_.__name__} 配置必须是映射")
    known = {f.name for f in fields(cls_)}
    unknown = set(data) - known
    kwargs = {k: v for k, v in data.items() if k in known}
    obj = cls_(**kwargs)
    if unknown:
        # 保留提示但不中断：方便配置渐进升级
        setattr(obj, "_unknown_keys", sorted(unknown))
    return obj


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on", "enabled")


def _expand_env(value: Any) -> Any:
    """递归展开字符串中的 ``${VAR}`` / ``${VAR:-default}``。"""
    if isinstance(value, str):
        def replace(match: "re.Match[str]") -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            return default if default is not None else match.group(0)

        try:
            expanded = _ENV_PATTERN.sub(replace, value)
            return os.path.expandvars(expanded)
        except Exception:  # pragma: no cover
            return value
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: Optional[str] = None) -> AgentConfig:
    """便捷函数：加载 Agent 配置。"""
    return AgentConfig.from_yaml(path)
