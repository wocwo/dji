"""逐句语义复核：用第二个 LLM 判断“引用片段是否真的支持这句话”。

词面 grounding 能抓数值/机型/无引用，但抓不住“指南针校准被用来解释高度显示”
这类语义不相关。这个模块把答案、问题、引用片段交给一个低温度 verifier，
要求它逐条返回 supported / partial / irrelevant / unsupported。

它默认关闭（额外一次 LLM 调用），在 DeepSeek / 生产配置中开启。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .llm import LLMClient
from .parser import extract_citation_refs
from .types import Citation, Message, Usage


SYSTEM_PROMPT = """你是严格的大疆产品手册事实核查员。
只根据给出的手册证据判断答案中的事实性 claim，不使用外部知识。
对每条 claim 输出一个 JSON 对象，最终只输出 JSON 数组，不要 Markdown：
[
  {"claim": "答案中的一句话", "verdict": "supported|partial|irrelevant|unsupported", "reason": "简短理由"}
]

判定标准：
- supported：证据直接支持该 claim；
- partial：证据只支持一部分，或数值/条件有差异；
- irrelevant：引用片段与该 claim 的主题无关（例如问高度显示，却引用指南针校准）；
- unsupported：证据没有提到该 claim。
只检查事实性内容；寒暄、建议、追问不需要检查。
无法确定时用 partial，不要猜。"""

_VERDICT_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "支持": "supported",
    "partial": "partial",
    "partially_supported": "partial",
    "部分支持": "partial",
    "irrelevant": "irrelevant",
    "无关": "irrelevant",
    "不相关": "irrelevant",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "不支持": "unsupported",
    "无支持": "unsupported",
}


@dataclass
class ClaimVerdict:
    claim: str
    verdict: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {"claim": self.claim, "verdict": self.verdict, "reason": self.reason}


@dataclass
class ClaimReport:
    checked: int = 0
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str = ""
    raw: str = ""

    @property
    def issues(self) -> List[ClaimVerdict]:
        return [verdict for verdict in self.verdicts if verdict.verdict != "supported"]

    @property
    def passed(self) -> bool:
        return not self.error and not self.issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "verdicts": [verdict.to_dict() for verdict in self.verdicts[:20]],
            "error": self.error,
            "raw": self.raw[:2000],
        }


class ClaimVerifier:
    """调用 verifier LLM 做逐句复核。"""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_claims: int = 12,
        max_answer_chars: int = 6000,
        max_evidence_chars: int = 12000,
    ):
        self.llm = llm
        self.max_claims = max(1, int(max_claims))
        self.max_answer_chars = max_answer_chars
        self.max_evidence_chars = max_evidence_chars

    def check(
        self,
        question: str,
        answer: str,
        citations: Sequence[Citation],
        evidence_texts: Dict[str, str],
    ) -> ClaimReport:
        refs = [ref.upper() for ref in extract_citation_refs(answer)]
        if not refs:
            return ClaimReport(error="答案没有引用，跳过语义复核")

        known = {str(citation.ref).upper(): citation for citation in citations}
        evidence_blocks: List[str] = []
        used_chars = 0
        for ref in refs:
            text = evidence_texts.get(ref, "")
            if not text:
                continue
            citation = known.get(ref)
            header = f"[{ref}]"
            if citation is not None:
                header += f" {citation.title or ''} 第{citation.page or '?'}页 {citation.section or ''}".strip()
            block = f"{header}\n{text.strip()}"
            if used_chars + len(block) > self.max_evidence_chars:
                break
            evidence_blocks.append(block)
            used_chars += len(block)

        if not evidence_blocks:
            return ClaimReport(error="没有可用的引用原文，跳过语义复核")

        prompt = (
            f"问题：\n{(question or '').strip()[:800]}\n\n"
            f"手册证据：\n{chr(10).join(evidence_blocks)}\n\n"
            f"待核查答案：\n{(answer or '').strip()[: self.max_answer_chars]}\n\n"
            f"最多检查 {self.max_claims} 条事实性 claim。只输出 JSON 数组。"
        )
        try:
            response = self.llm.complete(
                [Message("system", SYSTEM_PROMPT), Message("user", prompt)],
                temperature=0.0,
                max_tokens=2200,
                thinking=False,
            )
        except Exception as exc:  # verifier 失败不应影响主答案
            return ClaimReport(error=f"verifier 调用失败：{exc}")

        report = ClaimReport(usage=response.usage, raw=response.text or "")
        claims = self._parse(response.text)
        if claims is None:
            report.error = "verifier 返回的不是合法 JSON"
            return report

        for item in claims[: self.max_claims]:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            verdict = _VERDICT_ALIASES.get(
                str(item.get("verdict") or "").strip().lower(),
                "partial",
            )
            reason = str(item.get("reason") or "").strip()
            report.verdicts.append(ClaimVerdict(claim=claim, verdict=verdict, reason=reason))

        report.checked = len(report.verdicts)
        if not report.verdicts:
            report.error = "verifier 没有返回可用的 claim"
        return report

    @staticmethod
    def _parse(text: str) -> Optional[List[Dict[str, Any]]]:
        if not text:
            return None
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
                if isinstance(data, dict):
                    data = data.get("claims") or []
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        # DeepSeek 在 thinking 模式下可能在 JSON 前后夹带文字，或数组被截断；
        # 退化为逐个提取完整的 JSON object。
        objects: List[Dict[str, Any]] = []
        for match in re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL):
            try:
                item = json.loads(match.group(0))
            except Exception:
                continue
            if isinstance(item, dict) and "claim" in item:
                objects.append(item)
        return objects or None


__all__ = ["ClaimVerifier", "ClaimReport", "ClaimVerdict", "SYSTEM_PROMPT"]
