"""共享测试夹具：临时 JSONL 语料 + 可注入脚本化 LLM 的 Agent。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent import Agent, AgentConfig  # noqa: E402
from agent.llm import ScriptedClient  # noqa: E402
from agent.policy import AgentPolicy  # noqa: E402

SAMPLE_DOCS: List[Dict[str, Any]] = [
    {
        "chunk_id": "c-battery",
        "text": (
            "智能飞行电池在低温环境下使用时，应先预热电池至 10℃ 以上；"
            "低温环境下禁止立即充电，需等待电池温度回升。长期存放应将电量保持在 40%-60%，"
            "并每三个月充放电一次。"
        ),
        "doc_id": "mavic4pro-manual",
        "title": "DJI Mavic 4 Pro 用户手册",
        "section": "智能飞行电池",
        "page": 12,
        "model": "DJI Mavic 4 Pro",
        "type": "文本",
        "lang": "zh",
    },
    {
        "chunk_id": "c-firmware",
        "text": (
            "固件升级失败时，请检查网络连接，重新下载固件包并再次升级；"
            "若仍失败，可使用 DJI Assistant 2 进行离线升级。升级过程中不要断开电源，"
            "升级完成后飞行器会自动重启。"
        ),
        "doc_id": "mavic4pro-manual",
        "title": "DJI Mavic 4 Pro 用户手册",
        "section": "固件升级",
        "page": 30,
        "model": "DJI Mavic 4 Pro",
        "type": "步骤",
        "lang": "zh",
    },
    {
        "chunk_id": "c-pairing",
        "text": (
            "遥控器与飞行器对频：开启飞行器和遥控器，进入遥控器设置页面，选择对频，"
            "按提示长按电源按键，等待指示灯变为绿色即表示对频成功。"
        ),
        "doc_id": "rc2-manual",
        "title": "DJI RC 2 用户手册",
        "section": "遥控器对频",
        "page": 8,
        "model": "DJI RC 2",
        "type": "步骤",
        "lang": "zh",
    },
    {
        "chunk_id": "c-obstacle",
        "text": (
            "飞行器避障功能在光线不足、水面、玻璃幕墙等场景下可能失效，"
            "请保持谨慎并随时准备手动避让。安全飞行前请确认返航高度设置。"
        ),
        "doc_id": "air3s-manual",
        "title": "DJI Air 3S 用户手册",
        "section": "飞行安全",
        "page": 45,
        "model": "DJI Air 3S",
        "type": "警告",
        "lang": "zh",
    },
]


@pytest.fixture
def sample_corpus(tmp_path: Path) -> str:
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for doc in SAMPLE_DOCS:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return str(path)


@pytest.fixture
def make_agent(tmp_path: Path):
    """工厂夹具：``make_agent([...replies], ...) -> Agent``。"""
    agents: List[Agent] = []

    def factory(
        replies: Sequence[Any],
        *,
        corpus_path: Optional[str] = None,
        enabled_tools: Optional[List[str]] = None,
        max_steps: int = 6,
        plain_text_final: bool = True,
        memory: bool = False,
        trace: bool = False,
        policy: Optional[AgentPolicy] = None,
        allowed_tools: Optional[List[str]] = None,
        llm: Any = None,
        tool_mode: str = "text",
        evidence_guard: bool = True,
        evidence_auto_search: bool = True,
        claim_verifier: bool = False,
        fact_coverage: bool = False,
        engine: str = "legacy",
    ) -> Agent:
        config = AgentConfig()
        config.llm.provider = "scripted"
        config.llm.model = "test-model"
        config.llm.tool_mode = tool_mode
        config.agent.evidence_guard = evidence_guard
        config.agent.evidence_auto_search = evidence_auto_search
        config.agent.claim_verifier = claim_verifier
        config.agent.fact_coverage = fact_coverage
        config.agent.engine = engine
        config.agent.max_steps = max_steps
        config.agent.max_seconds = 30
        config.agent.retry_delay = 0.0
        config.agent.plain_text_final = plain_text_final
        config.knowledge.backend = "jsonl"
        config.knowledge.jsonl_path = corpus_path or str(tmp_path / "missing.jsonl")
        config.memory.enabled = memory
        config.memory.db_path = str(tmp_path / "memory.db")
        config.trace.enabled = trace
        config.trace.dir = str(tmp_path / "traces")
        config.tools.enabled = enabled_tools or [
            "manual_search",
            "manual_fetch",
            "calculator",
            "clock",
            "update_plan",
            "save_note",
            "recall_notes",
        ]
        client = llm or ScriptedClient(config.llm, replies, repeat_last=False)
        agent_policy = policy or AgentPolicy.from_config(config, allowed_tools=allowed_tools)
        agent = Agent(config, llm=client, policy=agent_policy)
        agents.append(agent)
        return agent

    yield factory

    for agent in agents:
        agent.close()
