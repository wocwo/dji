"""Agent 评测：用可复现的任务集衡量“会不会用工具、答得对不对、引用是否可靠”。

与 RAG 时代的检索指标（Recall/NDCG）不同，Agent 评测关注端到端行为：
- 期望调用的工具是否被调用；
- 最终答案是否包含关键事实；
- 是否给出来源引用；
- 状态、耗时与步数是否在预算内。

默认用例面向 DJI 手册演示语料，可用 ``python -m agent eval`` 运行。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .agent import Agent


@dataclass
class EvalCase:
    """一条评测用例。"""

    id: str
    task: str
    expected_tools: List[str] = field(default_factory=list)
    answer_contains: List[str] = field(default_factory=list)
    min_citations: int = 0
    expected_status: str = "ok"
    max_duration_ms: Optional[float] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_CASES: List[EvalCase] = [
    EvalCase(
        id="battery-care",
        task="智能飞行电池在低温环境下如何保养？",
        expected_tools=["manual_search"],
        answer_contains=["电池"],
        min_citations=1,
        tags=["knowledge", "battery"],
    ),
    EvalCase(
        id="firmware-issue",
        task="固件升级失败怎么办？",
        expected_tools=["manual_search"],
        answer_contains=["固件"],
        min_citations=1,
        tags=["knowledge", "firmware"],
    ),
    EvalCase(
        id="math",
        task="请计算 (5000-249)/1000",
        expected_tools=["calculator"],
        answer_contains=["4.751"],
        min_citations=0,
        tags=["utility", "math"],
    ),
]


def evaluate_case(agent: Agent, case: EvalCase) -> Dict[str, Any]:
    """运行单条用例并返回判定明细。"""
    started = time.time()
    result = agent.ask(case.task, persist=False)
    elapsed_ms = (time.time() - started) * 1000

    used_tools = [step.tool_call.name for step in result.steps if step.tool_call is not None]
    checks = {
        "status": result.status == case.expected_status,
        "tools": all(tool in used_tools for tool in case.expected_tools),
        "answer": all(
            keyword.lower() in (result.answer or "").lower()
            for keyword in case.answer_contains
        ),
        "citations": len(result.citations) >= case.min_citations,
        "duration": case.max_duration_ms is None or elapsed_ms <= case.max_duration_ms,
    }
    passed = all(checks.values())
    return {
        "id": case.id,
        "task": case.task,
        "passed": passed,
        "checks": checks,
        "status": result.status,
        "answer": result.answer,
        "used_tools": used_tools,
        "expected_tools": case.expected_tools,
        "citation_count": len(result.citations),
        "steps": len(result.steps),
        "duration_ms": round(elapsed_ms, 1),
        "trace_id": result.trace_id,
        "warnings": result.warnings,
        "tags": case.tags,
    }


def run_eval(
    agent: Agent,
    cases: Optional[Sequence[EvalCase]] = None,
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """运行评测集并汇总指标。"""
    cases = list(cases or DEFAULT_CASES)
    started = time.time()
    records: List[Dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        if verbose:
            print(f"[{index}/{len(cases)}] {case.id}: {case.task}")
        record = evaluate_case(agent, case)
        records.append(record)
        if verbose:
            mark = "PASS" if record["passed"] else "FAIL"
            print(
                f"  {mark} · {record['status']} · {record['duration_ms']:.0f}ms · "
                f"tools={record['used_tools']} · citations={record['citation_count']}"
            )
            if not record["passed"]:
                failed = [name for name, ok in record["checks"].items() if not ok]
                print(f"  未通过检查项: {failed}")

    total = len(records)
    passed = sum(1 for r in records if r["passed"])
    tool_ok = sum(
        1 for r in records if r["checks"].get("tools")
    )
    answer_ok = sum(1 for r in records if r["checks"].get("answer"))
    citation_ok = sum(1 for r in records if r["checks"].get("citations"))
    durations = [r["duration_ms"] for r in records]

    report = {
        "total": total,
        "passed": passed,
        "success_rate": round(passed / total, 4) if total else 0.0,
        "tool_accuracy": round(tool_ok / total, 4) if total else 0.0,
        "answer_accuracy": round(answer_ok / total, 4) if total else 0.0,
        "citation_accuracy": round(citation_ok / total, 4) if total else 0.0,
        "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
        "total_duration_ms": round((time.time() - started) * 1000, 1),
        "provider": getattr(getattr(agent, "llm", None), "name", ""),
        "cases": records,
    }
    return report


def format_report(report: Dict[str, Any]) -> str:
    """把评测报告渲染为文本。"""
    lines = [
        "=" * 62,
        "Agent 评测报告",
        "=" * 62,
        f"用例数      : {report['total']}",
        f"通过        : {report['passed']} ({report['success_rate'] * 100:.1f}%)",
        f"工具准确率  : {report['tool_accuracy'] * 100:.1f}%",
        f"答案命中率  : {report['answer_accuracy'] * 100:.1f}%",
        f"引用覆盖率  : {report['citation_accuracy'] * 100:.1f}%",
        f"平均耗时    : {report['avg_duration_ms']:.0f}ms",
        "-" * 62,
    ]
    for case in report["cases"]:
        mark = "PASS" if case["passed"] else "FAIL"
        lines.append(f"[{mark}] {case['id']}  status={case['status']}  {case['duration_ms']:.0f}ms")
        lines.append(f"       答案: {case['answer'][:100]}")
    lines.append("=" * 62)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


__all__ = ["EvalCase", "DEFAULT_CASES", "evaluate_case", "run_eval", "format_report", "save_report"]
