"""页面语料合并测试。"""

from __future__ import annotations

import json

from scripts.merge_page_corpora import merge


def _write(path, items):
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def test_pdf_pages_replace_same_doc_fallback(tmp_path):
    pdf = tmp_path / "pdf.jsonl"
    chunk = tmp_path / "chunk.jsonl"
    output = tmp_path / "out.jsonl"
    _write(
        pdf,
        [
            {
                "chunk_id": "pdf-1",
                "doc_id": "dji_mavic_4_pro_user_manual",
                "page": 49,
                "model": "DJI Mavic 4 Pro",
                "text": "PDF 页面正文",
            }
        ],
    )
    _write(
        chunk,
        [
            {
                "chunk_id": "old-1",
                "doc_id": "dji_mavic_4_pro_user_manual",
                "page": 0,
                "model": "DJI Mavic 4 Pro",
                "text": "旧 chunk 页面",
            },
            {
                "chunk_id": "other-1",
                "doc_id": "other_manual",
                "page": 2,
                "model": "DJI Other",
                "text": "其它文档",
            },
        ],
    )
    stats = merge(pdf, chunk, output)
    items = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert stats["pdf_records"] == 1
    assert {item["chunk_id"] for item in items} == {"pdf-1", "other-1"}
    assert all(item.get("source_kind") for item in items)
