# -*- coding: utf-8 -*-
"""Simple test script"""
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime
import os

print("=" * 60)
print("DJI Vector DB Build Test")
print("=" * 60)

# Load chunks
print("\n[1] Loading chunks...")
chunks = []
count = 0
max_c = 500
with open('dji_chunks_final.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if count >= max_c:
            break
        if line.strip():
            chunks.append(json.loads(line))
            count += 1
print(f"  Loaded {len(chunks)} chunks")

# Load embeddings
print("\n[2] Loading embeddings...")
cache = np.load('embed_cache.npz', allow_pickle=True)
emb_list = cache['embeddings'][:max_c].tolist()
print(f"  Loaded {len(emb_list)} embeddings, dim={len(emb_list[0])}")

# Prepare data
print("\n[3] Preparing data...")
ids = []
docs = []
metas = []
for i, chunk in enumerate(chunks):
    doc_id = chunk.get('doc_id', 'unknown')
    cid = f"{doc_id}_chunk_{i}"
    text = chunk.get('text', '') or chunk.get('content', '')
    ids.append(cid)
    docs.append(text[:500] if text else '')
    metas.append({
        'doc_id': doc_id,
        'model': chunk.get('model', ''),
        'type': chunk.get('type', 'text'),
        'lang': chunk.get('lang', 'unknown')
    })
print(f"  Prepared {len(ids)} records")

# Build index
print("\n[4] Building ChromaDB index...")
client = chromadb.PersistentClient(
    path='./chroma_final',
    settings=Settings(anonymized_telemetry=False)
)

try:
    client.delete_collection('uav_manual_db')
except:
    pass

coll = client.create_collection(
    name='uav_manual_db',
    metadata={'hnsw:space': 'cosine'}
)

# Add in batches
bs = 100
for i in range(0, len(ids), bs):
    end = min(i + bs, len(ids))
    coll.add(
        embeddings=emb_list[i:end],
        ids=ids[i:end],
        documents=docs[i:end],
        metadatas=metas[i:end]
    )
    print(f"  Added {end}/{len(ids)}")

print(f"\n  Total vectors: {coll.count()}")

# Save stats
stats = {
    "total_vectors": coll.count(),
    "collection": "uav_manual_db",
    "distance_metric": "cosine",
    "embedding_dim": len(emb_list[0])
}
with open("index_stat.json", "w") as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
