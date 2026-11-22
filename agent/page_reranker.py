"""页面级 rerank：对检索候选页重新排序。

BM25 偏好“词重复多、页面长”，但正确页经常是定义页/规格页，排在第 2–8。
这里用可解释特征做第二段排序：

- BM25 归一化分数
- 查询实体覆盖率
- 标题/章节覆盖率
- 数值/单位覆盖
- 定义页/规格页意图
- 机型精确匹配
- 文档类型与目录页惩罚

不调用 LLM，可离线调参。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:℃|°C|m/s|km/h|km|cm|mm|m|g|kg|min|h|秒|分钟|小时|天|个月|年|米|公里|%|mAh|Wh|V|A|W|dB|fps|MP|GB|MB|ms)")


def _norm(text: str) -> str:
    text = (text or "").lower().replace("°c", "℃")
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", "", text)


def _numbers(text: str) -> Set[str]:
    return {re.sub(r"\s+", "", match.group(0)) for match in _NUMBER_RE.finditer(text or "")}


def _terms(text: str, limit: int = 12) -> List[str]:
    try:
        from .knowledge import extract_terms
        return [term for term in extract_terms(text, max_terms=limit) if len(term) >= 2]
    except Exception:
        return []


@dataclass
class PageFeatures:
    base: float = 0.0
    terms: float = 0.0
    section: float = 0.0
    numbers: float = 0.0
    definition: float = 0.0
    model: float = 0.0
    page_type: float = 0.0
    penalty: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "base": round(self.base, 3),
            "terms": round(self.terms, 3),
            "section": round(self.section, 3),
            "numbers": round(self.numbers, 3),
            "definition": round(self.definition, 3),
            "model": round(self.model, 3),
            "page_type": round(self.page_type, 3),
            "penalty": round(self.penalty, 3),
            "score": round(self.score, 3),
        }


class PageReranker:
    """可解释的规则 rerank，输入 (bm25_score, Passage) 列表。"""

    def __init__(self, *, pool: int = 30):
        self.pool = max(4, int(pool))

    def rank(self, query: str, model: str, candidates: Sequence[Tuple[float, Any]]) -> List[Tuple[float, Any]]:
        if not candidates:
            return []
        pool = list(candidates[: self.pool])
        max_base = max((score for score, _ in pool), default=1.0) or 1.0
        q_terms = _terms(query, limit=14)
        q_norm = _norm(query)
        q_numbers = _numbers(query)
        query_model = (model or "").lower().removeprefix("dji ").strip()
        want_definition = any(word in query for word in ("是什么", "包括", "哪些", "怎么", "如何", "工作", "原理"))
        want_spec = any(word in query for word in ("规格", "参数", "尺寸", "重量", "容量", "范围"))

        ranked: List[Tuple[float, Any]] = []
        for base, passage in pool:
            text = passage.text or ""
            normalized_text = _norm(text)
            section_title = _norm(f"{getattr(passage, 'section', '')} {getattr(passage, 'title', '')}")
            page_terms = [term for term in q_terms if term in text]
            term_cov = len(page_terms) / len(q_terms) if q_terms else 0.0
            section_cov = (
                sum(1 for term in q_terms if term in section_title) / len(q_terms)
                if q_terms else 0.0
            )
            page_numbers = _numbers(text)
            number_cov = (
                len(q_numbers & page_numbers) / len(q_numbers) if q_numbers else 0.0
            )

            definition = 0.0
            if "视觉系统" in q_norm and "视觉系统" in normalized_text:
                if any(word in normalized_text for word in ("摄像头", "位于", "分别")):
                    definition = 1.0
            if "返航点" in q_norm and "返航点" in normalized_text:
                if "记录" in normalized_text or "gnss" in normalized_text:
                    definition = max(definition, 1.0)
            if "充电温度" in q_norm and (
                "充电温度" in normalized_text or normalized_text.count("℃") >= 2
            ):
                definition = max(definition, 0.9)
            if "对频" in q_norm and "对频" in normalized_text:
                definition = max(definition, 0.8)

            page_model = str(getattr(passage, "model", "") or "").lower()
            if query_model and page_model:
                if query_model in page_model:
                    model_score = 1.0
                else:
                    core = query_model.removeprefix("dji ").strip()
                    family = " ".join(core.split()[:2])
                    model_score = 0.5 if family and family in page_model else 0.0
            else:
                model_score = 0.5

            section = str(getattr(passage, "section", "") or "")
            doc_type = str((getattr(passage, "metadata", {}) or {}).get("doc_type") or "")
            page_type = 0.5
            if want_spec and any(word in section_title for word in ("规格", "参数", "技术规格")):
                page_type = 1.0
            elif any(word in doc_type for word in ("user_manual", "user_guide")):
                page_type = 0.8
            elif "safety" in doc_type:
                page_type = 0.6
            elif "release_notes" in doc_type:
                page_type = 0.3
            if (getattr(passage, "metadata", {}) or {}).get("type") == "table" and want_spec:
                page_type = max(page_type, 0.9)

            penalty = 0.0
            page = str(getattr(passage, "page", ""))
            if page == "0" or "目录" in text[:120] or "阅读提示" in text[:120]:
                penalty += 0.25
            if "release_notes" in doc_type and not want_spec:
                penalty += 0.1

            base_norm = max(0.0, min(1.0, base / max_base))
            score = (
                0.90 * base_norm
                + 0.04 * term_cov
                + 0.02 * section_cov
                + 0.02 * number_cov
                + 0.01 * definition
                + 0.01 * model_score
                - penalty
            )
            if want_definition and definition > 0:
                score *= 1.05
            ranked.append((score, passage))

        ranked.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        return ranked


__all__ = ["PageReranker", "PageFeatures"]
