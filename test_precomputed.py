"""测试预计算嵌入模式 - 简化版本"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime

print("="*60)
print("DJI 向量数据库构建 - 预计算嵌入模式")
print("="*60)

# 1. 加载数据
print("\n[1] 加载数据")
chunks = []
max_chunks = 1000
with open('dji_chunks_final.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx >= max_chunks:
            break
        if line.strip():
            chunks.append(json.loads(line))
print(f"  加载了 {len(chunks)} 个文本块")

# 2. 加载预计算嵌入
print("\n[2] 加载预计算嵌入")
cache = np.load('embed_cache.npz', allow_pickle=True)
cache_ids = cache['ids'][:max_chunks]
cache_embeddings = cache['embeddings'][:max_chunks]
print(f"  加载了 {len(cache_embeddings)} 个向量")
print(f"  向量维度: {cache_embeddings.shape[1]}")

# 3. 准备数据
print("\n[3] 准备数据")
ids = []
documents = []
metadatas = []

for i, chunk in enumerate(chunks):
    doc_id = chunk.get('doc_id', 'unknown')
    chunk_id = f"{doc_id}_chunk_{i}"
    text = chunk.get('text', '') or chunk.get('content', '')
    
    ids.append(chunk_id)
    documents.append(text[:1000] if text else '')
    metadatas.append({
        'doc_id': doc_id,
        'chunk_id': chunk.get('chunk_id', ''),
        'model': chunk.get('model', ''),
        'title': chunk.get('title', ''),
        'type': chunk.get('type', '文本'),
        'lang': chunk.get('lang', 'unknown'),
        'page': chunk.get('page', -1)
    })

embeddings = cache_embeddings.tolist()
print(f"  准备了 {len(ids)} 条记录")

# 4. 构建ChromaDB索引
print("\n[4] 构建ChromaDB索引")
client = chromadb.PersistentClient(
    path='./chroma_final',
    settings=Settings(anonymized_telemetry=False)
)

# 删除旧集合（如果存在）
try:
    client.delete_collection('uav_manual_db')
except:
    pass

collection = client.create_collection(
    name='uav_manual_db',
    metadata={'hnsw:space': 'cosine'}
)
print(f"  创建集合: uav_manual_db")

# 分批添加
batch_size = 500
total = len(ids)
for i in range(0, total, batch_size):
    end = min(i + batch_size, total)
    collection.add(
        embeddings=embeddings[i:end],
        ids=ids[i:end],
        documents=documents[i:end],
        metadatas=metadatas[i:end]
    )
    print(f"  已添加 {end}/{total}")

print(f"  索引完成，共 {collection.count()} 个向量")

# 5. 保存统计
stats = {
    "total_vectors": collection.count(),
    "collection_name": "uav_manual_db",
    "distance_metric": "cosine",
    "embedding_dim": len(embeddings[0]),
    "last_update": datetime.now().isoformat()
}
with open("index_stat.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 6. 验证
print("\n[5] 基础验证")
print(f"  向量总数: {collection.count()}")

# 获取样本
sample = collection.get(limit=3)
print(f"  样本ID: {sample['ids'][:3]}")
print(f"  元数据字段: {list(sample['metadatas'][0].keys()) if sample['metadatas'] else []}")

print("\n" + "="*60)
print("✅ 构建完成！")
print("="*60)
print("\n产出文件:")
print("  - chroma_final/          向量数据库文件")
print("  - index_stat.json        索引统计")
