"""轻量 RSI 查询策略：Curriculum → Actor(retrieval) → Verifier → Memory。

不是 RSIAgent 的完整 OSWorld/ALE 运行时，而是在当前手册 Agent 上复现它的
核心闭环：从失败/弱案例探索查询策略，验证后写入可复用记忆，再在盲测集上
冻结使用。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from agent.fact_coverage import FactCoverageChecker  # noqa: E402
from agent.knowledge import JsonlKnowledgeBase  # noqa: E402

VARIANTS = ["", " 使用注意事项", " 规格参数", " 用户手册", " 参数 表"]


def _rank(kb, question, model, expected_docs, expected_pages):
    results = kb.search(question, top_k=8, model=model or None)
    for index, passage in enumerate(results, start=1):
        if (not expected_docs or passage.doc_id in expected_docs) and int(passage.page or 0) in expected_pages:
            return index
    return 99


def main() -> int:
    kb = JsonlKnowledgeBase("data/agent_manual_kb.jsonl")
    checker = FactCoverageChecker()
    train = [json.loads(line) for line in Path("data/eval_golden.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [json.loads(line) for line in Path("data/eval_holdout_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    scores = defaultdict(lambda: defaultdict(list))
    for case in train:
        pages = {int(p) for p in case.get("expected_pages") or []}
        if not pages:
            continue
        intent = checker.detect(case["question"]) or "generic"
        docs = set(case.get("expected_doc_ids") or [])
        for variant in VARIANTS:
            rank = _rank(kb, case["question"] + variant, case.get("model"), docs, pages)
            scores[intent][variant].append(rank)

    memory: Dict[str, str] = {}
    for intent, variants in scores.items():
        baseline = variants.get("", [])
        if not baseline:
            continue
        base_recall1 = sum(1 for rank in baseline if rank == 1) / len(baseline)
        base_mean = sum(baseline) / len(baseline)
        best = min(
            variants,
            key=lambda variant: (
                sum(1 for rank in variants[variant] if rank == 1) / len(variants[variant]),
                -sum(variants[variant]) / len(variants[variant]),
            ),
        )
        candidate = variants[best]
        cand_recall1 = sum(1 for rank in candidate if rank == 1) / len(candidate)
        cand_mean = sum(candidate) / len(candidate)
        # 只有训练集上明确优于 baseline 才写入记忆；否则不学习，避免负迁移
        if cand_recall1 > base_recall1 + 0.05 or (
            abs(cand_recall1 - base_recall1) < 1e-9 and cand_mean < base_mean - 0.5
        ):
            memory[intent] = best
    Path("data/rsi_query_memory.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate(use_memory: bool) -> Dict[str, Any]:
        ranks = []
        for case in test:
            pages = {int(p) for p in case.get("expected_pages") or []}
            if not pages:
                continue
            intent = checker.detect(case["question"]) or "generic"
            suffix = memory.get(intent, "") if use_memory else ""
            ranks.append(_rank(kb, case["question"] + suffix, case.get("model"), set(case.get("expected_doc_ids") or []), pages))
        total = len(ranks)
        return {
            "total": total,
            "recall@1": round(sum(1 for rank in ranks if rank == 1) / total, 4) if total else 0,
            "recall@4": round(sum(1 for rank in ranks if rank <= 4) / total, 4) if total else 0,
            "mrr": round(sum(1 / rank for rank in ranks if rank < 99) / total, 4) if total else 0,
        }

    report = {"memory": memory, "baseline": evaluate(False), "rsi": evaluate(True)}
    Path("data/rsi_eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
