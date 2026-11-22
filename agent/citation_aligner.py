"""引用页对齐：答案已有引用，但可能引到了相邻/相关页。

检索阶段 p60/p61/p63 都进了 evidence；模型可能只在答案里写 p60。
这里对每个带引用的句子，在同一文档的候选页里重新计算词面/数值支持度，
若另一页明显更支持该句，则把该句引用替换到更具体的页。

只在同 doc_id 内对齐，避免跨机型替换；阈值较高，避免把正确引用改坏。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set, Tuple

from .grounding import extract_citation_refs, tokenize
from .types import Citation

_SENTENCE_RE = re.compile(r"(?<=[。！？!?])(?!\s*[\[［])\s*|\n+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:℃|°C|m/s|km/h|km|cm|mm|m|g|kg|min|h|秒|分钟|小时|天|个月|年|米|公里|%|mAh|Wh|V|A|W|dB|fps|MP|GB|MB|ms)")


def _numbers(text: str) -> Set[str]:
    return {re.sub(r"\s+", "", match.group(0)) for match in _NUMBER_RE.finditer(text or "")}


def _score(sentence: str, text: str) -> float:
    sentence_tokens = {token for token in tokenize(sentence) if len(token) >= 2}
    text_tokens = {token for token in tokenize(text) if len(token) >= 2}
    if not sentence_tokens:
        return 0.0
    overlap = len(sentence_tokens & text_tokens) / len(sentence_tokens)
    sentence_numbers = _numbers(sentence)
    if sentence_numbers:
        numeric = len(sentence_numbers & _numbers(text)) / len(sentence_numbers)
        return overlap + 0.5 * numeric
    return overlap


@dataclass
class CitationChange:
    sentence: str
    before: List[str]
    after: str
    score_before: float
    score_after: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sentence": self.sentence[:200],
            "before": list(self.before),
            "after": self.after,
            "score_before": round(self.score_before, 3),
            "score_after": round(self.score_after, 3),
        }


@dataclass
class CitationAlignment:
    answer: str
    changes: List[CitationChange] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"changes": [change.to_dict() for change in self.changes]}


class CitationAligner:
    def __init__(self, *, min_gain: float = 0.12):
        self.min_gain = min_gain

    def align(
        self,
        answer: str,
        citations: Sequence[Citation],
        evidence_texts: Dict[str, str],
    ) -> CitationAlignment:
        by_ref = {str(citation.ref).upper(): citation for citation in citations}
        candidates = [
            (ref.upper(), citation, evidence_texts.get(ref.upper(), ""))
            for ref, citation in by_ref.items()
            if evidence_texts.get(ref.upper())
        ]
        if not candidates:
            return CitationAlignment(answer=answer)

        pieces = [part for part in _SENTENCE_RE.split(answer or "") if part and part.strip()]
        changes: List[CitationChange] = []
        output: List[str] = []
        for piece in pieces:
            refs = [ref.upper() for ref in extract_citation_refs(piece)]
            if len(refs) != 1:
                output.append(piece)
                continue
            current_ref = refs[0]
            current = by_ref.get(current_ref)
            if current is None:
                output.append(piece)
                continue
            doc_candidates = [
                (ref, citation, text)
                for ref, citation, text in candidates
                if citation.doc_id and citation.doc_id == current.doc_id
            ]
            if len(doc_candidates) <= 1:
                output.append(piece)
                continue
            scores: List[Tuple[float, str]] = []
            for ref, _citation, text in doc_candidates:
                scores.append((_score(piece, text), ref))
            scores.sort(reverse=True)
            best_score, best_ref = scores[0]
            current_score = next((score for score, ref in scores if ref == current_ref), 0.0)
            if best_ref != current_ref and best_score > current_score + self.min_gain:
                new_piece = re.sub(
                    r"[\[［]\s*S\d+\s*[\]］]",
                    f"[{best_ref}]",
                    piece,
                    count=1,
                    flags=re.IGNORECASE,
                )
                output.append(new_piece)
                changes.append(
                    CitationChange(
                        sentence=piece.strip(),
                        before=[current_ref],
                        after=best_ref,
                        score_before=current_score,
                        score_after=best_score,
                    )
                )
            else:
                output.append(piece)
        return CitationAlignment(answer="\n".join(output).strip(), changes=changes)


__all__ = ["CitationAligner", "CitationAlignment", "CitationChange"]
