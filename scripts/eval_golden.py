"""高质量评测：检索命中、引用命中、claim/数值覆盖、拒答/澄清行为。"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from agent import Agent  # noqa: E402


def norm(text: str) -> str:
    text = (text or "").lower().replace("°c", "℃").replace(" ", "")
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    text = text.replace("公里", "km").replace("米", "m")
    for chinese, number in (
        ("十", "10"), ("九", "9"), ("八", "8"), ("七", "7"),
        ("六", "6"), ("五", "5"), ("四", "4"), ("三", "3"),
        ("两", "2"), ("二", "2"), ("一", "1"),
    ):
        text = text.replace(chinese, number)
    text = text.replace("\u00a0", "").replace("·", "").replace("，", ",")
    return text


def page_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def extract_evidence(result: Any) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    for step in getattr(result, "steps", []) or []:
        data = getattr(getattr(step, "result", None), "data", {}) or {}
        for passage in data.get("passages") or []:
            if isinstance(passage, dict):
                evidence.append(passage)
    return evidence


def evaluate_case(agent: Agent, case: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    result = agent.ask(case["question"], persist=False)
    elapsed = (time.time() - started) * 1000
    answer = result.answer or ""
    normalized = norm(answer)
    evidence = extract_evidence(result)
    citations = result.citations or []

    expected_docs = set(case.get("expected_doc_ids") or [])
    expected_pages = {page_int(page) for page in (case.get("expected_pages") or [])}
    expected_pages.discard(None)
    expected_claims = case.get("expected_claims") or []
    expected_numbers = case.get("expected_numbers") or []
    behavior = case.get("expected_behavior", "answer")

    def hit(item: Dict[str, Any]) -> bool:
        doc_id = str(item.get("doc_id") or "")
        page = page_int(item.get("page"))
        doc_ok = not expected_docs or doc_id in expected_docs
        page_ok = not expected_pages or page in expected_pages
        return doc_ok and page_ok

    retrieval_hit = any(hit(item) for item in evidence) if expected_pages else None
    citation_hit = any(
        hit({"doc_id": c.doc_id, "page": c.page}) for c in citations
    ) if expected_pages else None
    claims_hit = all(norm(claim) in normalized for claim in expected_claims)
    numbers_hit = all(norm(number) in normalized for number in expected_numbers)
    required_any = case.get("required_any") or []
    behavior_ok = (
        any(norm(word) in normalized for word in required_any)
        if behavior in ("refusal", "clarify")
        else True
    )
    forbidden_hit = any(norm(item) in normalized for item in (case.get("must_not_claim") or []))

    checks = {
        "status": result.status == "ok",
        "retrieval": retrieval_hit if retrieval_hit is not None else True,
        "citation": citation_hit if citation_hit is not None else True,
        "claims": claims_hit,
        "numbers": numbers_hit,
        "behavior": behavior_ok,
        "forbidden": not forbidden_hit,
    }
    return {
        "id": case["id"],
        "question": case["question"],
        "model": case.get("model", ""),
        "tags": case.get("tags", []),
        "passed": all(checks.values()),
        "checks": checks,
        "status": result.status,
        "answer": answer,
        "retrieval_hit": retrieval_hit,
        "citation_hit": citation_hit,
        "expected_docs": sorted(expected_docs),
        "expected_pages": sorted(expected_pages),
        "evidence": [
            {"doc_id": item.get("doc_id"), "page": item.get("page")}
            for item in evidence[:12]
        ],
        "citation_count": len(citations),
        "steps": len(result.steps),
        "duration_ms": round(elapsed, 1),
        "tokens": result.usage.total_tokens if result.usage else 0,
        "warnings": result.warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="高质量手册评测")
    parser.add_argument("--cases", default="data/eval_golden.jsonl")
    parser.add_argument("--config", default="configs/agent.deepseek.yaml")
    parser.add_argument("--output", default="data/eval_baseline.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="逗号分隔的 case id")
    args = parser.parse_args()

    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = {item.strip() for item in args.ids.split(",") if item.strip()}
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit:
        cases = cases[: args.limit]

    agent = Agent.from_config(args.config)
    records = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
        try:
            record = evaluate_case(agent, case)
        except Exception as exc:
            record = {
                "id": case["id"], "question": case["question"], "passed": False,
                "error": f"{type(exc).__name__}: {exc}", "checks": {},
            }
        records.append(record)
        mark = "PASS" if record.get("passed") else "FAIL"
        print(
            f"  {mark} status={record.get('status')} "
            f"retrieval={record.get('retrieval_hit')} citation={record.get('citation_hit')} "
            f"claims={record.get('checks', {}).get('claims')} "
            f"numbers={record.get('checks', {}).get('numbers')} "
            f"{record.get('duration_ms', 0):.0f}ms",
            flush=True,
        )

    eligible = [r for r in records if r.get("expected_pages")]
    total = len(records)
    summary = {
        "total": total,
        "passed": sum(1 for r in records if r.get("passed")),
        "pass_rate": round(sum(1 for r in records if r.get("passed")) / total, 4) if total else 0,
        "retrieval_recall": round(
            sum(1 for r in eligible if r.get("retrieval_hit")) / len(eligible), 4
        ) if eligible else 0,
        "citation_recall": round(
            sum(1 for r in eligible if r.get("citation_hit")) / len(eligible), 4
        ) if eligible else 0,
        "claim_accuracy": round(
            sum(1 for r in records if r.get("checks", {}).get("claims")) / total, 4
        ) if total else 0,
        "number_accuracy": round(
            sum(1 for r in records if r.get("checks", {}).get("numbers")) / total, 4
        ) if total else 0,
        "avg_duration_ms": round(sum(r.get("duration_ms", 0) for r in records) / total, 1) if total else 0,
        "total_tokens": sum(r.get("tokens", 0) for r in records),
        "elapsed_s": round(time.time() - started, 1),
        "provider": getattr(agent.llm, "name", ""),
        "cases": records,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False, indent=2))
    print(f"报告: {args.output}")
    agent.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
