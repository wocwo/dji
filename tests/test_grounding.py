"""引用规范化与 grounding 校验测试。"""

from __future__ import annotations

from agent.grounding import GroundingVerifier, extract_models, normalize_citations
from agent.types import Citation


def _citation() -> Citation:
    return Citation(
        ref="S1",
        chunk_id="c1",
        doc_id="dji_mavic_4_pro_user_manual",
        title="DJI Mavic 4 Pro 用户手册",
        section="智能飞行电池",
        page=50,
        model="DJI Mavic 4 Pro",
    )


def test_normalize_bare_citations():
    assert normalize_citations("见 S1S2S5") == "见 [S1][S2][S5]"
    assert normalize_citations("见 S1、S2、S5") == "见 [S1][S2][S5]"
    assert normalize_citations("见 [S1][S2]") == "见 [S1][S2]"
    assert normalize_citations("Mavic S1 电池") == "Mavic S1 电池"


def test_supported_sentence_passes():
    verifier = GroundingVerifier()
    report = verifier.check(
        "在低温环境下，建议飞行前将电池预热至 10℃以上。[S1]",
        [_citation()],
        {"S1": "在低温环境下，建议在飞行前将电池预热至 10℃以上，预热至 20℃以上更佳。"},
    )
    assert report.passed
    assert report.issues == []
    assert report.supported_sentences == 1
    assert report.coverage == 1.0


def test_numeric_mismatch_detected():
    verifier = GroundingVerifier()
    report = verifier.check(
        "建议飞行前将电池预热至 99℃以上。[S1]",
        [_citation()],
        {"S1": "在低温环境下，建议在飞行前将电池预热至 10℃以上。"},
    )
    assert not report.passed
    assert any(issue.kind == "numeric_mismatch" for issue in report.issues)


def test_uncited_fact_detected():
    verifier = GroundingVerifier()
    report = verifier.check(
        "电池应该预热到 10℃。",
        [_citation()],
        {"S1": "在低温环境下，建议将电池预热至 10℃以上。"},
    )
    assert any(issue.kind == "uncited" for issue in report.issues)
    assert report.coverage == 0.0


def test_model_scope_detected():
    verifier = GroundingVerifier()
    report = verifier.check(
        "Mini 4 Pro 建议预热到 5℃以上。[S1]",
        [_citation()],
        {"S1": "在低温环境下，建议将电池预热至 10℃以上。"},
    )
    assert any(issue.kind == "model_scope" for issue in report.issues)


def test_extract_models():
    models = extract_models("Mavic 4 Pro 与 Mini 4 Pro 的低温建议不同")
    assert any("Mavic 4 Pro" in model for model in models)
    assert any("Mini 4 Pro" in model for model in models)
    assert extract_models("RC 2 遥控器怎么对频") == ["RC 2"]
    assert extract_models("DJI RC-N3 怎么充电") == ["DJI RC-N3"]
    assert extract_models("Goggles 3 怎么用") == ["Goggles 3"]


def test_no_citations_scores_neutral():
    verifier = GroundingVerifier()
    report = verifier.check("你好，有什么可以帮你？", [], {})
    assert report.cited_sentences == 0
    assert report.score == 1.0
