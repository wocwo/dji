"""可复现的端到端评测：读取 JSONL 用例，输出 JSON + Markdown。

用法：
    python scripts/run_eval.py --cases data/eval_holdout_v2.jsonl \
        --config configs/agent.deepseek.yaml \
        --output reports/eval_holdout_2026-11-22.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import Agent, AgentConfig
from agent.policy import AgentPolicy


def _norm(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("°", "°").replace("℃", "°c").replace(" ", "")
    text = text.replace("－", "-").replace("–", "-")
    return re.sub(r"\s+", "", text)


def _has(answer: str, expected: str) -> bool:
    return _norm(expected) in _norm(answer)


def _citation_hit(result: Any, case: Dict[str, Any]) -> bool:
    docs = set(case.get("expected_doc_ids") or [])
    pages = set(case.get("expected_pages") or [])
    for citation in result.citations or []:
        if docs and citation.doc_id and citation.doc_id not in docs:
            continue
        if pages and citation.page not in pages:
            continue
        return True
    return False


def run(cases_path: str, config_path: str, output: str) -> int:
    config = AgentConfig.from_yaml(config_path)
    config.knowledge.backend = "jsonl"
    config.knowledge.jsonl_path = "data/agent_manual_kb.jsonl"
    config.memory.enabled = False
    agent = Agent(config, policy=AgentPolicy.from_config(config))
    rows: List[Dict[str, Any]] = []
    started = time.time()
    try:
        for line in Path(cases_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            t0 = time.time()
            result = agent.ask(case["question"], persist=False)
            answer = result.answer or ""
            claims = case.get("expected_claims") or []
            numbers = case.get("expected_numbers") or []
            checks = {
                "status": result.status == "ok",
                "citations": _citation_hit(result, case) or (not case.get("expected_behavior") == "answer"),
                "claims": all(_has(answer, item) for item in claims),
                "numbers": all(_has(answer, item) for item in numbers),
            }
            passed = all(checks.values())
            usage = result.usage
            if dataclasses.is_dataclass(usage):
                usage = dataclasses.asdict(usage)
            row = {
                "id": case["id"],
                "question": case["question"],
                "passed": passed,
                "checks": checks,
                "status": result.status,
                "answer": answer,
                "citations": [
                    {"ref": c.ref, "doc_id": c.doc_id, "page": c.page, "title": c.title}
                    for c in (result.citations or [])
                ],
                "citation_count": len(result.citations or []),
                "duration_ms": round((time.time() - t0) * 1000, 1),
                "usage": usage,
                "warnings": result.warnings,
                "tags": case.get("tags") or [],
            }
            rows.append(row)
            print(f"[{len(rows):02d}] {'PASS' if passed else 'FAIL'} {case['id']} {row['duration_ms']}ms {checks}", flush=True)
    finally:
        agent.close()

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    citations_ok = sum(1 for row in rows if row["checks"]["citations"])
    claims_ok = sum(1 for row in rows if row["checks"]["claims"])
    numbers_ok = sum(1 for row in rows if row["checks"]["numbers"])
    report = {
        "cases": cases_path,
        "config": config_path,
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "citation_rate": round(citations_ok / total, 4) if total else 0,
        "claim_rate": round(claims_ok / total, 4) if total else 0,
        "number_rate": round(numbers_ok / total, 4) if total else 0,
        "avg_duration_ms": round(sum(row["duration_ms"] for row in rows) / total, 1) if total else 0,
        "elapsed_sec": round(time.time() - started, 1),
        "failures": [row for row in rows if not row["passed"]],
        "rows": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{passed}/{total} passed · citation {citations_ok}/{total} · claim {claims_ok}/{total} · avg {report['avg_duration_ms']}ms")
    print(f"saved: {output}")
    return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/eval_holdout_v2.jsonl")
    parser.add_argument("--config", default="configs/agent.deepseek.yaml")
    parser.add_argument("--output", default="reports/eval_holdout.json")
    args = parser.parse_args()
    return run(args.cases, args.config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
