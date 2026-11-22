"""引用页对齐测试。"""

from __future__ import annotations

from agent.citation_aligner import CitationAligner
from agent.types import Citation


def _cite(ref: str, page: int) -> Citation:
    return Citation(
        ref=ref,
        chunk_id=f"c{page}",
        doc_id="mini4-manual",
        title="DJI Mini 4 Pro 用户手册",
        page=page,
        model="DJI Mini 4 Pro",
    )


def test_aligns_to_better_same_doc_page():
    aligner = CitationAligner()
    result = aligner.align(
        "电池每隔 3 个月左右重新充电一次。[S1]",
        [_cite("S1", 60), _cite("S2", 61)],
        {
            "S1": "电池指示灯和电量显示说明。",
            "S2": "长期存放时，每隔 3 个月左右重新充电一次以保持电池活性。",
        },
    )
    assert result.changes
    assert "[S2]" in result.answer
    assert "[S1]" not in result.answer


def test_keeps_best_citation():
    aligner = CitationAligner()
    result = aligner.align(
        "电池每隔 3 个月左右重新充电一次。[S1]",
        [_cite("S1", 61), _cite("S2", 60)],
        {
            "S1": "长期存放时，每隔 3 个月左右重新充电一次以保持电池活性。",
            "S2": "电池指示灯和电量显示说明。",
        },
    )
    assert not result.changes
    assert result.answer.endswith("[S1]")


def test_does_not_cross_documents():
    aligner = CitationAligner()
    other = Citation(ref="S2", chunk_id="c2", doc_id="air3s-manual", page=61)
    result = aligner.align(
        "电池每隔 3 个月左右重新充电一次。[S1]",
        [_cite("S1", 60), other],
        {
            "S1": "电池指示灯说明。",
            "S2": "长期存放时，每隔 3 个月左右重新充电一次。",
        },
    )
    assert not result.changes
