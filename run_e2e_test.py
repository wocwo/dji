"""
最小端到端检索测试：create_pipeline().search(query)，将结果打印/收集为 JSON。
"""

import traceback
import sys

print("[1] starting import")
from rag_pipeline import create_pipeline
print("[2] import done")

import json
print("[3] creating pipeline")
pipeline = create_pipeline()
print("[4] pipeline ready")

queries = [
    "Mavic 4 Pro max flight altitude",
]

print("[5] beginning queries")
results = []
for q in queries:
    print("[6] searching:", q)
    try:
        r = pipeline.search(q)
        print("[7] search done, formatting")
        results.append({
            "query": q,
            "primary_intent": r.context.intent.primary_intent.value,
            "secondary_intent": r.context.intent.secondary_intent.value,
            "recall_chunks": r.multipath_result.total_unique_chunks,
            "output_chunks": r.rerank_result.output_count,
            "time_ms": r.total_time_ms,
            "assembled_chunks": r.generation_context.assembled_context.chunk_count if r.generation_context else 0,
            "context_chars": r.generation_context.assembled_context.total_chars if r.generation_context else 0,
            "prompt_chars": r.generation_context.prompt_metadata.get("prompt_chars", 0) if r.generation_context else 0,
        })
        print("[8] formatted")
    except Exception as e:
        print("[ERROR] during search:", str(e))
        traceback.print_exc()
        sys.exit(1)

with open("e2e_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("[9] All done. Output saved to e2e_result.json")
