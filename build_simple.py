#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI 向量数据库构建 - 简化版
"""

import os
import sys
import json
import shutil
import numpy as np
import chromadb

# 清理旧数据
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# 参数
max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 100
print(f"处理 {max_chunks} 条数据...")

# 1. 加载数据
chunks = []
with open("dji_chunks_final.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= max_chunks:
            break
        chunks.append(json.loads(line))
print(f"加载了 {len(chunks)} 条")

# 2. 加载嵌入
cache = np.load("embed_cache.npz", allow_pickle=True)
embeddings = cache["embeddings"][:max_chunks]
print(f"加载了 {len(embeddings)} 个向量, 维度 {embeddings.shape[1]}")

# 3. 准备简单数据
ids = [f"chunk_{i}" for i in range(len(chunks))]
docs = []
metas = []
for i, chunk in enumerate(chunks):
    text = chunk.get("text", "") or chunk.get("content", "")
    docs.append(text[:500] if text else "")
    # 简化元数据，避免复杂类型
    metas.append({
        "doc_id": str(chunk.get("doc_id", "")),
        "page": str(chunk.get("page", "-1"))
    })

# 4. 创建ChromaDB - 最简配置
client = chromadb.PersistentClient(path="./chroma_db")

coll = client.create_collection(
    name="manual_db",
    metadata={"hnsw:space": "cosine"}
)
print("创建集合成功")

# 5. 分批添加 - 小批次
batch_size = 50
for i in range(0, len(ids), batch_size):
    end = min(i + batch_size, len(ids))
    coll.add(
        embeddings=embeddings[i:end].tolist(),
        ids=ids[i:end],
        documents=docs[i:end],
        metadatas=metas[i:end]
    )
    print(f"  添加 {end}/{len(ids)}")

print(f"\n完成! 共 {coll.count()} 个向量")
print(f"数据目录: ./chroma_db")
