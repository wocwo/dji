"""引用规范化与逐句 grounding 校验。

高星文档问答项目的共同点不是“让模型多写引用”，而是：
1. 引用格式必须可解析（[S1]，而不是 S1S2S5）；
2. 每个事实句最好有引用；
3. 引用片段必须能支持这句话（至少数值/关键实体一致）；
4. 无法支持时降级为“未验证/推断”，而不是继续当作事实。

本模块只做**轻量、确定性**的检查，不替代人工审核：
- 不判断深层语义是否等价；
- 但能抓出无引用、数值不一致、跨机型、引用与句子无词面交集等常见问题。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .knowledge import tokenize
from .parser import extract_citation_refs


# 常见的 S1S2S5 / S1、S2 / ［S1］ 变体
_BARE_RUN_RE = re.compile(
    r"(?<![A-Za-z0-9\[［])(S\d+(?:\s*[、,，]?\s*S\d+)+)(?![A-Za-z0-9\]］])",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:℃|°C|m/s|km/h|km|cm|mm|m|g|kg|min|h|秒|分钟|小时|天|个月|年|米|公里|%|mAh|Wh|V|A|W|dB|fps|MP|GB|MB|ms)"
)
_MODEL_RE = re.compile(
    r"\b(?:DJI\s+)?(?:(?:Mavic|Mini|Air|Matrice|Avata|Inspire|Phantom|Neo|Flip|Ronin|Osmo)\s*[0-9A-Za-z-]*\s*(?:Pro|Mini|Enterprise|Cine|Classic|T|E|M|S|RTK)?|RC\s*(?:Pro|-?\s*N?\d+)|Goggles\s*(?:3|2|V2)?)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])(?!\s*[\[［])\s*|\n+")


def normalize_citations(text: str) -> str:
    """把模型常见的裸引用 ``S1S2S5`` / ``S1、S2`` 规范为 ``[S1][S2][S5]``。"""
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        refs = [f"S{number}".upper() for number in re.findall(r"S(\d+)", match.group(1), re.IGNORECASE)]
        return "".join(f"[{ref}]" for ref in refs)

    return _BARE_RUN_RE.sub(replace, text)


def extract_models(text: str) -> List[str]:
    models: List[str] = []
    for match in _MODEL_RE.finditer(text or ""):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value and value.lower() not in (m.lower() for m in models):
            models.append(value)
    return models


def _numbers(text: str) -> Set[str]:
    return {re.sub(r"\s+", "", match.group(0)) for match in _NUMBER_RE.finditer(text or "")}


def _citation_labels(citations: Sequence[Any]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for citation in citations:
        ref = str(getattr(citation, "ref", "") or "")
        if not ref:
            continue
        labels[ref.upper()] = " ".join(
            str(value or "")
            for value in (
                getattr(citation, "title", ""),
                getattr(citation, "doc_id", ""),
                getattr(citation, "model", ""),
                getattr(citation, "section", ""),
            )
        )
    return labels


def _is_non_claim(sentence: str) -> bool:
    """标题、结论摘要、推断/澄清句不强制逐句引用。"""
    text = sentence.strip()
    if not text:
        return True
    if re.match(r"^#{1,6}\s+", text):
        return True
    if re.fullmatch(r"\*{0,2}[一二三四五六七八九十]+、.{0,40}\*{0,2}", text):
        return True
    if text.startswith("**") and text.endswith("**") and len(text) <= 60:
        return True
    if re.match(r"^\*{0,2}(结论|总结|说明|注意|前言)\*{0,2}\s*[:：]", text):
        return True
    if any(marker in text for marker in ("推断", "待确认", "请提供", "请告知", "你未提供", "确认后", "请补充", "告诉我")):
        return True
    if text.endswith(("？", "?")) and any(marker in text for marker in ("请问", "您", "你", "机型", "型号")):
        return True
    return False


@dataclass
class GroundingIssue:
    """一条 grounding 问题。"""

    kind: str  # uncited / weak_support / numeric_mismatch / model_scope / malformed
    sentence: str
    detail: str
    refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "sentence": self.sentence[:240],
            "detail": self.detail,
            "refs": list(self.refs),
        }


@dataclass
class GroundingReport:
    """一篇答案的 grounding 报告。"""

    score: float
    coverage: float
    checked_sentences: int
    cited_sentences: int
    supported_sentences: int
    uncited_sentences: int = 0
    issues: List[GroundingIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(issue.kind in ("numeric_mismatch", "model_scope") for issue in self.issues)

    def to_dict(self) -> Dict[str, object]:
        return {
            "score": round(self.score, 4),
            "coverage": round(self.coverage, 4),
            "checked_sentences": self.checked_sentences,
            "cited_sentences": self.cited_sentences,
            "supported_sentences": self.supported_sentences,
            "uncited_sentences": self.uncited_sentences,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class GroundingVerifier:
    """逐句检查答案是否被引用片段支持。"""

    def __init__(self, *, min_overlap: float = 0.18):
        self.min_overlap = min_overlap

    def check(
        self,
        answer: str,
        citations: Sequence[Any],
        evidence_texts: Dict[str, str],
        *,
        question: str = "",
    ) -> GroundingReport:
        labels = _citation_labels(citations)
        sentences = [part.strip(" -\t•◆●▪·") for part in _SENTENCE_SPLIT_RE.split(answer or "") if part.strip()]
        issues: List[GroundingIssue] = []
        checked = 0
        cited_count = 0
        supported = 0
        uncited_count = 0

        for sentence in sentences:
            if len(sentence) < 8:
                continue
            checked += 1
            refs = [ref.upper() for ref in extract_citation_refs(sentence)]
            if not refs:
                # 只有带数字/单位的句子才算“事实句”；标题/推断/澄清句不强行要求引用
                if _is_non_claim(sentence):
                    continue
                if _numbers(sentence) or any(term in sentence for term in ("手册", "固件", "电池", "返航", "校准")):
                    uncited_count += 1
                    issues.append(
                        GroundingIssue(
                            kind="uncited",
                            sentence=sentence,
                            detail="句子包含事实性数值/主题词，但没有引用",
                        )
                    )
                continue

            cited_count += 1
            refs = [ref for ref in refs if ref in labels or ref in evidence_texts]
            if not refs:
                issues.append(
                    GroundingIssue(
                        kind="malformed",
                        sentence=sentence,
                        detail="引用的 [Sx] 不在本轮检索结果中",
                    )
                )
                continue

            texts = [evidence_texts.get(ref, "") for ref in refs]
            combined = " ".join(texts)
            labels_text = " ".join(labels.get(ref, "") for ref in refs)

            sentence_tokens = {token for token in tokenize(sentence) if len(token) >= 2}
            evidence_tokens = {token for token in tokenize(combined + " " + labels_text) if len(token) >= 2}
            overlap = (
                len(sentence_tokens & evidence_tokens) / len(sentence_tokens)
                if sentence_tokens
                else 1.0
            )

            sentence_numbers = _numbers(sentence)
            evidence_numbers = _numbers(combined)
            missing_numbers = sorted(sentence_numbers - evidence_numbers)
            if missing_numbers:
                issues.append(
                    GroundingIssue(
                        kind="numeric_mismatch",
                        sentence=sentence,
                        detail=f"数值未出现在被引片段中：{', '.join(missing_numbers)}",
                        refs=refs,
                    )
                )
            elif overlap < self.min_overlap:
                issues.append(
                    GroundingIssue(
                        kind="weak_support",
                        sentence=sentence,
                        detail=f"句子与引用片段的词面重合度较低（{overlap:.0%}）",
                        refs=refs,
                    )
                )
            else:
                supported += 1

            sentence_models = extract_models(sentence)
            if sentence_models:
                cited_text = (combined + " " + labels_text).lower()
                missing_models = [model for model in sentence_models if model.lower() not in cited_text]
                if missing_models:
                    issues.append(
                        GroundingIssue(
                            kind="model_scope",
                            sentence=sentence,
                            detail=f"句子提到 {', '.join(missing_models)}，但引用来源未出现该机型",
                            refs=refs,
                        )
                    )

        score = supported / cited_count if cited_count else 1.0
        total_facts = cited_count + uncited_count
        coverage = cited_count / total_facts if total_facts else 1.0
        return GroundingReport(
            score=score,
            coverage=coverage,
            checked_sentences=checked,
            cited_sentences=cited_count,
            supported_sentences=supported,
            uncited_sentences=uncited_count,
            issues=issues,
        )


__all__ = [
    "GroundingVerifier",
    "GroundingReport",
    "GroundingIssue",
    "normalize_citations",
    "extract_models",
]
