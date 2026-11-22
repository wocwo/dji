"""证据闸门：判断“这条回答能不能直接交付”，以及何时触发 RAG 回退。

背景：
- 手册事实类问题，不能只靠模型记忆直接回答；
- 但模型经常在第一步就输出一个看起来很像答案的 final；
- 仅靠 LLM 自评“我答得对不对”不可靠，且多一次模型调用更慢。

这里用确定性规则做第一道闸门：
1. 先判断问题是否属于“手册事实/高风险”；
2. 如果模型直接给出 final，但本轮没有任何成功的手册检索证据，则拒绝该答案；
3. 运行时自动执行一次 ``manual_search``，把 observation 回灌，让模型重新作答；
4. 若仍无证据（知识库不可用/查不到），保留答案但标记 warning，避免静默伪造。

它不是“判断答案事实是否正确”的万能校验器，而是防止**无证据直接作答**的护栏。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .knowledge import extract_model_phrase
from .parser import extract_citation_refs


# 手册事实相关词。命中即认为需要证据；宁可多检索，不要凭记忆答安全/参数问题。
_MANUAL_TERMS = {
    "电池", "智能飞行电池", "充电", "续航", "保养", "固件", "升级", "固件回退",
    "遥控器", "对频", "起飞", "降落", "返航", "避障", "图传", "存储卡",
    "桨叶", "螺旋桨", "电机", "指南针", "校准", "IMU", "RTK", "飞行限制",
    "禁飞", "限高", "返航高度", "智能跟随", "拍照", "录像", "云台", "相机",
    "激活", "绑定", "参数", "规格", "安全", "警告", "故障", "报错", "错误码",
    "说明书", "手册", "机型", "保修", "机场", "司空", "调参", "App",
}

# 安全/参数类高风险问题：即使模型给了答案，也必须先有手册证据。
_HIGH_RISK_TERMS = {
    "电池", "充电", "固件", "升级", "返航", "避障", "飞行限制", "禁飞",
    "限高", "桨叶", "螺旋桨", "安全", "警告", "故障", "错误码",
}

_GENERAL_PATTERNS = [
    re.compile(r"^[\d\s\+\-\*/%\(\)\.\^×÷]+$"),
    re.compile(r"^(你好|您好|hi|hello|嗨|谢谢|在吗|你是谁|介绍一下你自己)[！!。.\s]*$", re.IGNORECASE),
    re.compile(r"(现在|今天|当前).{0,8}(几点|几号|日期|时间|星期)"),
    re.compile(r"^(讲个笑话|写首诗|写个故事)$"),
]


def _is_general(question: str) -> bool:
    text = question.strip()
    if any(pattern.search(text) for pattern in _GENERAL_PATTERNS):
        return True
    # 纯算术/换算：有数字、有运算符、有“计算/等于/换算”等词
    if any(word in text for word in ("计算", "等于", "换算", "乘以", "除以")):
        if re.search(r"\d", text) and re.search(r"[+\-*/%^×÷]", text):
            return True
    return False


def _is_clarification(answer: str) -> bool:
    """允许模型先反问机型/版本，而不是被迫检索一个模糊问题。"""
    text = answer.strip()
    if len(text) > 120:
        return False
    if not ("？" in text or "?" in text):
        return False
    return any(
        word in text
        for word in ("请问", "您能", "请提供", "哪个机型", "哪一款", "什么型号", "型号")
    )


@dataclass
class QuestionProfile:
    """问题侧的风险画像。"""

    requires_evidence: bool
    risk: str  # high / normal / general
    reason: str
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "requires_evidence": self.requires_evidence,
            "risk": self.risk,
            "reason": self.reason,
            "matched_terms": list(self.matched_terms),
        }


@dataclass
class AnswerCheck:
    """答案侧的检查结果。"""

    passed: bool
    needs_retrieval: bool
    issues: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    unknown_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "passed": self.passed,
            "needs_retrieval": self.needs_retrieval,
            "issues": list(self.issues),
            "refs": list(self.refs),
            "unknown_refs": list(self.unknown_refs),
        }


class EvidenceGate:
    """规则驱动的证据闸门。"""

    def __init__(self, *, enabled: bool = True, auto_search: bool = True):
        self.enabled = enabled
        self.auto_search = auto_search

    def profile(self, question: str) -> QuestionProfile:
        text = (question or "").strip()
        matched: List[str] = []
        for term in sorted(_MANUAL_TERMS, key=len, reverse=True):
            if term in text:
                matched.append(term)
        model = extract_model_phrase(text)
        if model and model not in matched:
            matched.append(model)

        general = _is_general(text)
        high_risk = any(term in text for term in _HIGH_RISK_TERMS)
        # 高风险问题即使表述成“算一下”，也不能绕过手册证据
        requires = bool(matched) and (not general or high_risk)
        if any(word in text for word in ("手册", "说明书", "官方")):
            requires = True

        if not requires:
            risk = "general"
            reason = "未命中手册事实词，允许直接回答"
        elif high_risk:
            risk = "high"
            reason = f"命中高风险手册词：{', '.join(matched[:6])}"
        else:
            risk = "normal"
            reason = f"命中手册领域词：{', '.join(matched[:6])}"
        return QuestionProfile(
            requires_evidence=requires,
            risk=risk,
            reason=reason,
            matched_terms=matched,
        )

    def check(
        self,
        profile: QuestionProfile,
        answer: str,
        *,
        known_refs: Sequence[str],
        evidence_count: int,
    ) -> AnswerCheck:
        """检查一条 final 答案是否可以有证据地交付。"""
        refs = extract_citation_refs(answer or "")
        known = {str(ref).upper() for ref in known_refs}
        unknown = [ref for ref in refs if ref not in known]
        issues: List[str] = []

        needs_retrieval = (
            self.enabled
            and profile.requires_evidence
            and evidence_count <= 0
            and not _is_clarification(answer or "")
        )
        if needs_retrieval:
            issues.append("missing_evidence")
        if evidence_count > 0 and not refs:
            issues.append("missing_citations")
        if unknown:
            issues.append("unknown_citations")

        return AnswerCheck(
            passed=not needs_retrieval,
            needs_retrieval=needs_retrieval,
            issues=issues,
            refs=refs,
            unknown_refs=unknown,
        )


__all__ = ["EvidenceGate", "QuestionProfile", "AnswerCheck"]
