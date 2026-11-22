"""知识库测试：分词、词项提取、BM25 排序与取回。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.errors import KnowledgeBaseError
from agent.knowledge import (
    JsonlKnowledgeBase,
    Passage,
    extract_model_phrase,
    extract_terms,
    tokenize,
)


def test_extract_model_phrase():
    assert extract_model_phrase("Mavic 4 Pro 电池低温保养") == "Mavic 4 Pro"
    assert extract_model_phrase("mini 4 pro 固件升级") == "Mini 4 Pro"
    assert extract_model_phrase("Air 3S 怎么对频") == "Air 3S"
    assert extract_model_phrase("电池怎么保养") is None


def test_tokenize_chinese_bigrams():
    tokens = tokenize("智能飞行电池")
    assert "智能" in tokens
    assert "能飞" in tokens
    assert "电池" in tokens


def test_tokenize_ascii():
    tokens = tokenize("DJI Mavic 4 Pro")
    assert "dji" in tokens
    assert "mavic" in tokens
    assert "pro" in tokens


def test_extract_terms_prefers_models():
    terms = extract_terms("Mavic 4 Pro 电池 低温 保养", max_terms=5)
    assert "Mavic" in terms
    assert "Pro" in terms
    assert "电池" in terms
    assert "保养" in terms


def test_extract_terms_chinese_question():
    terms = extract_terms("智能飞行电池怎么保养？", max_terms=5)
    assert "智能飞行电池" in terms
    assert "保养" in terms


def test_extract_terms_empty():
    assert extract_terms("") == []
    assert extract_terms("的了吗呢") == []


def test_jsonl_search_ranking(sample_corpus):
    kb = JsonlKnowledgeBase(sample_corpus)
    results = kb.search("电池 低温 保养", top_k=2)
    assert results
    assert results[0].chunk_id == "c-battery"
    assert results[0].score > 0
    assert "低温" in results[0].text


def test_jsonl_search_no_match(sample_corpus):
    kb = JsonlKnowledgeBase(sample_corpus)
    assert kb.search("量子计算机", top_k=3) == []


def test_jsonl_fetch(sample_corpus):
    kb = JsonlKnowledgeBase(sample_corpus)
    passage = kb.fetch("c-firmware")
    assert passage is not None
    assert "固件" in passage.text
    assert kb.fetch("missing") is None


def test_jsonl_stats(sample_corpus):
    kb = JsonlKnowledgeBase(sample_corpus)
    stats = kb.stats()
    assert stats["backend"] == "jsonl-bm25"
    assert stats["count"] == 4


def test_jsonl_missing_file(tmp_path):
    kb = JsonlKnowledgeBase(str(tmp_path / "nope.jsonl"))
    with pytest.raises(KnowledgeBaseError):
        kb.search("电池")


def test_jsonl_array_format(tmp_path):
    path = tmp_path / "array.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "a1",
                    "text": "返航高度应在起飞前设置，默认返航高度为 100 米。",
                    "title": "飞行手册",
                    "page": 3,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    kb = JsonlKnowledgeBase(str(path))
    results = kb.search("返航高度", top_k=1)
    assert results and results[0].chunk_id == "a1"


def test_passage_helpers():
    passage = Passage(
        chunk_id="c1",
        text="a\nb",
        metadata={"title": "无标题", "page": "3", "section": "电池保养", "doc_id": "d1"},
    )
    assert passage.title == ""
    assert passage.page == 3
    assert passage.section == "电池保养"
    assert passage.doc_id == "d1"
    assert passage.preview(1) == "a…"


@pytest.mark.skipif(
    not Path("chroma_final").exists(),
    reason="需要本地 chroma_final 向量库",
)
def test_chroma_optional_smoke():
    """可选集成测试：只在向量库存在时运行，验证真实词法检索。"""
    from agent.knowledge import ChromaKnowledgeBase

    kb = ChromaKnowledgeBase("./chroma_final", "uav_manual_db")
    results = kb.search("智能飞行电池", top_k=2)
    assert results
    assert all(passage.text for passage in results)
    assert kb.fetch(results[0].chunk_id) is not None
