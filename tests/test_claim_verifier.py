"""逐句语义复核（ClaimVerifier）测试。"""

from __future__ import annotations

import json

from agent.claim_verifier import ClaimVerifier
from agent.llm import LLMResponse
from agent.types import Citation, Usage


class FakeLLM:
    def __init__(self, text: str = "", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMResponse(
            text=self.text,
            usage=Usage(prompt_tokens=100, completion_tokens=20),
        )


def _citation() -> Citation:
    return Citation(
        ref="S1",
        chunk_id="c1",
        doc_id="dji-mavic-4-pro",
        title="DJI Mavic 4 Pro 用户手册",
        section="智能飞行电池",
        page=50,
        model="DJI Mavic 4 Pro",
    )


def test_verifier_parses_issues():
    payload = [
        {"claim": "指南针校准可以修复高度显示。", "verdict": "irrelevant", "reason": "指南针影响航向，不是高度来源"},
        {"claim": "低温下电池需要预热。", "verdict": "supported", "reason": "片段直接支持"},
    ]
    verifier = ClaimVerifier(FakeLLM(json.dumps(payload, ensure_ascii=False)))
    report = verifier.check(
        "高度显示不对怎么办？",
        "指南针校准可以修复高度显示。[S1] 低温下电池需要预热。[S1]",
        [_citation()],
        {"S1": "智能飞行电池在低温下应先预热至 10℃以上；指南针校准步骤见安全设置。"},
    )
    assert report.checked == 2
    assert report.passed is False
    assert [issue.verdict for issue in report.issues] == ["irrelevant"]


def test_verifier_accepts_code_fence():
    payload = [{"claim": "电池需要预热", "verdict": "supported", "reason": "ok"}]
    verifier = ClaimVerifier(FakeLLM("```json\n" + json.dumps(payload) + "\n```"))
    report = verifier.check("电池", "电池需要预热。[S1]", [_citation()], {"S1": "电池需要预热。"})
    assert report.passed is True
    assert report.checked == 1


def test_verifier_malformed_json_reports_error():
    verifier = ClaimVerifier(FakeLLM("这不是 JSON"))
    report = verifier.check("电池", "电池需要预热。[S1]", [_citation()], {"S1": "电池需要预热。"})
    assert report.error
    assert report.checked == 0


def test_verifier_without_citations_skips():
    verifier = ClaimVerifier(FakeLLM("[]"))
    report = verifier.check("电池", "没有引用", [], {})
    assert report.error
    assert verifier.llm.calls == 0


def test_verifier_llm_failure_reports_error():
    verifier = ClaimVerifier(FakeLLM(error=RuntimeError("boom")))
    report = verifier.check("电池", "电池需要预热。[S1]", [_citation()], {"S1": "电池需要预热。"})
    assert "verifier 调用失败" in report.error
