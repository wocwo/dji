"""
手动分步调试 RAG 检索：导入流水线、处理查询、单独测语义通道等。
"""

import sys
import traceback

print("[1] import")
from rag_pipeline import create_pipeline
p = create_pipeline()
print("[2] pipeline ready")

q = "Mavic 4 Pro max flight altitude"
ctx = p.query_system.process_query(q)
print("[3] query processed")

# Test semantic only
print("[4] semantic retrieve")
try:
    ch_semantic = p.multipath_retriever.retrieve_semantic(
        ctx.enhanced.precise_query,
        where_filter=ctx.filter_condition.chroma_filter if ctx.filter_condition else None,
        top_k=10
    )
    print("[4] semantic ok, results:", len(ch_semantic.results))
except Exception as e:
    print("[4] semantic failed:", e)
    traceback.print_exc()

# Test keyword
print("[5] keyword retrieve")
try:
    ch_keyword = p.multipath_retriever.retrieve_keyword(ctx.enhanced.precise_query, top_k=10)
    print("[5] keyword ok, results:", len(ch_keyword.results))
except Exception as e:
    print("[5] keyword failed:", e)
    traceback.print_exc()

# Test metadata
print("[6] metadata retrieve")
try:
    ch_metadata = p.multipath_retriever.retrieve_metadata(
        ctx.enhanced.precise_query, ctx.intent, top_k=10
    )
    print("[6] metadata ok, results:", len(ch_metadata.results))
except Exception as e:
    print("[6] metadata failed:", e)
    traceback.print_exc()

# Test expanded
print("[7] expanded retrieve")
try:
    ch_expanded = p.multipath_retriever.retrieve_expanded(
        ctx.enhanced.expanded_queries, top_k=5
    )
    print("[7] expanded ok, results:", len(ch_expanded.results))
except Exception as e:
    print("[7] expanded failed:", e)
    traceback.print_exc()

print("[8] all steps finished")
