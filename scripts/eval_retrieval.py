"""只评测检索层：正确 doc/page 是否进 top-k，以及排名。不调用 LLM。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from agent.config import load_config  # noqa: E402
from agent.knowledge import create_knowledge_base  # noqa: E402


def page_int(value: Any) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="检索层盲测")
    parser.add_argument("--cases", default="data/eval_holdout_v2.jsonl")
    parser.add_argument("--config", default="configs/agent.deepseek.yaml")
    parser.add_argument("--output", default="data/eval_retrieval.json")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    kb = create_knowledge_base(config.knowledge)
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    records: List[Dict[str, Any]] = []

    for case in cases:
        expected_docs = set(case.get("expected_doc_ids") or [])
        expected_pages = {page_int(p) for p in (case.get("expected_pages") or [])}
        expected_pages.discard(None)
        if not expected_pages:
            continue
        results = kb.search(case["question"], top_k=args.top_k, model=case.get("model") or None)
        rank = None
        for index, passage in enumerate(results, start=1):
            doc_ok = not expected_docs or passage.doc_id in expected_docs
            page_ok = page_int(passage.page) in expected_pages
            if doc_ok and page_ok:
                rank = index
                break
        records.append(
            {
                "id": case["id"],
                "question": case["question"],
                "model": case.get("model"),
                "expected_docs": sorted(expected_docs),
                "expected_pages": sorted(expected_pages),
                "rank": rank,
                "top": [
                    {"doc_id": item.doc_id, "page": item.page, "type": item.metadata.get("type", "")}
                    for item in results
                ],
            }
        )

    total = len(records)
    summary = {
        "total": total,
        "recall@1": round(sum(1 for r in records if r["rank"] == 1) / total, 4) if total else 0,
        "recall@4": round(sum(1 for r in records if r["rank"] and r["rank"] <= 4) / total, 4) if total else 0,
        "recall@8": round(sum(1 for r in records if r["rank"] and r["rank"] <= 8) / total, 4) if total else 0,
        "mrr": round(sum(1 / r["rank"] for r in records if r["rank"]) / total, 4) if total else 0,
        "rerank_calls": getattr(kb, "rerank_calls", 0),
        "rerank_skipped": getattr(kb, "rerank_skipped", 0),
        "misses": [r for r in records if not r["rank"]],
        "late_hits": [r for r in records if r["rank"] and r["rank"] > 4],
        "cases": records,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, ensure_ascii=False, indent=2))
    print(f"报告: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
