"""证据闸门测试：问题画像、无证据拦截与引用诊断。"""

from __future__ import annotations

from agent.verifier import EvidenceGate


def _profile(question: str):
    return EvidenceGate().profile(question)


def test_manual_question_requires_evidence():
    profile = _profile("Mavic 4 Pro 智能飞行电池低温下怎么保养？")
    assert profile.requires_evidence is True
    assert profile.risk == "high"
    assert "电池" in profile.matched_terms


def test_firmware_question_requires_evidence():
    profile = _profile("固件升级失败怎么办？")
    assert profile.requires_evidence is True
    assert profile.risk == "high"


def test_general_question_does_not_require_evidence():
    assert _profile("你好").requires_evidence is False
    assert _profile("今天几号").requires_evidence is False
    assert _profile("请计算 (5000-249)/1000").requires_evidence is False


def test_manual_mention_overrides_arithmetic():
    profile = _profile("请计算电池的充电功率")
    assert profile.requires_evidence is True


def test_clarification_is_allowed_without_evidence():
    gate = EvidenceGate()
    profile = gate.profile("电池低温怎么保养？")
    check = gate.check(
        profile,
        "请问您的具体机型是哪一款？不同机型预热温度不同。",
        known_refs=[],
        evidence_count=0,
    )
    assert check.needs_retrieval is False


def test_missing_evidence_blocks_answer():
    gate = EvidenceGate()
    profile = gate.profile("固件升级失败怎么办？")
    check = gate.check(
        profile,
        "重新升级即可，不需要查手册。",
        known_refs=[],
        evidence_count=0,
    )
    assert check.needs_retrieval is True
    assert "missing_evidence" in check.issues


def test_answer_with_evidence_and_unknown_ref():
    gate = EvidenceGate()
    profile = gate.profile("固件升级失败怎么办？")
    check = gate.check(
        profile,
        "请重新下载固件。[S9]",
        known_refs=["S1"],
        evidence_count=1,
    )
    assert check.needs_retrieval is False
    assert "unknown_citations" in check.issues
    assert check.unknown_refs == ["S9"]


def test_gate_disabled_does_not_block():
    gate = EvidenceGate(enabled=False)
    profile = gate.profile("电池怎么保养？")
    check = gate.check(profile, "凭经验回答", known_refs=[], evidence_count=0)
    assert check.needs_retrieval is False
