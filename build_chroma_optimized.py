#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI 向量数据库构建 - ChromaDB优化版（适合大规模数据）
优化策略:
    1. 更大的批次写入（减少事务开销）
    2. 禁用wal模式提升写入速度
    3. 先构建小索引再增量添加
    
使用:
    python build_chroma_optimized.py [chunk数量]
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime

CHROMA_PATH = "./chroma_optimized"
BATCH_SIZE = 1000  # 更大的批次


def build_large_index(max_chunks: int):
    """构建大规模索引"""
    print(f"构建 {max_chunks:,} 条数据的优化索引...")
    
    # 清理旧数据
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # 加载数据
    print("\n[1/3] 加载数据...")
    chunks = []
    with open("dji_chunks_final.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_chunks:
                break
            chunks.append(json.loads(line))
    
    cache = np.load("embed_cache.npz", allow_pickle=True)
    embeddings = cache["embeddings"][:len(chunks)]
    
    print(f"      加载了 {len(chunks):,} 条数据")
    
    # 准备数据
    print("\n[2/3] 准备数据...")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    docs = []
    metas = []
    
    for chunk in chunks:
        text = chunk.get("text", "") or chunk.get("content", "")
        docs.append(text[:300] if text else "")
        metas.append({
            "doc_id": str(chunk.get("doc_id", ""))[:50],
            "page": str(chunk.get("page", -1))
        })
    
    # 创建客户端 - 优化配置
    print("\n[3/3] 构建索引...")
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            # 尝试优化SQLite性能
            chroma_segment_cache_policy=" LRU"
        )
    )
    
    # 创建集合
    coll = client.create_collection(
        name="manual_db",
        metadata={"hnsw:space": "cosine"}
    )
    
    # 分批添加 - 优化进度显示
    total = len(ids)
    start_time = time.time()
    
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        
        coll.add(
            embeddings=embeddings[i:end].tolist(),
            ids=ids[i:end],
            documents=docs[i:end],
            metadatas=metas[i:end]
        )
        
        # 进度和预估
        if i > 0:
            elapsed = time.time() - start_time
            speed = i / elapsed
            remaining = (total - i) / speed
            print(f"      {end:,}/{total:,} | 速度: {speed:.0f}条/秒 | 剩余: {remaining/60:.1f}分钟")
    
    total_time = time.time() - start_time
    
    print(f"\n✅ 完成!")
    print(f"   总时间: {total_time/60:.1f} 分钟")
    print(f"   平均速度: {total/total_time:.0f} 条/秒")
    print(f"   向量总数: {coll.count()}")
    print(f"   数据目录: {CHROMA_PATH}")
    
    # 保存统计
    stats = {
        "total_vectors": coll.count(),
        "build_time_minutes": round(total_time/60, 2),
        "speed_per_second": round(total/total_time, 1),
        "created_at": datetime.now().isoformat()
    }
    with open("build_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    return coll


if __name__ == "__main__":
    max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    build_large_index(max_chunks)
