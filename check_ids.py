"""
调试脚本：对比 embed_cache.npz 中的 id 与 dji_chunks_final.jsonl 前若干条 chunk_id。
"""

import json
import numpy as np

# 检查缓存中的ID
data = np.load('embed_cache.npz', allow_pickle=True)
cache_ids = data['ids'][:10]
print('Cache IDs (first 10):')
for id_ in cache_ids:
    print(f'  {id_}')

# 检查chunks中的ID
print('\nChunk IDs (first 10):')
with open('dji_chunks_final.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        chunk = json.loads(line)
        print(f"  {chunk.get('chunk_id', 'N/A')}")
