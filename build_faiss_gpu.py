#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI 向量数据库构建 - FAISS GPU加速版
构建167万向量推荐使用GPU，速度提升5-10倍

要求:
    pip install faiss-gpu  # 有NVIDIA GPU
    或
    pip install faiss-cpu  # 仅CPU

使用:
    python build_faiss_gpu.py [chunk数量] [use_gpu]
    
示例:
    python build_faiss_gpu.py 1000000 1   # 100万条，使用GPU
    python build_faiss_gpu.py 10000 0     # 1万条，仅CPU
"""

import os
import sys
import json
import time
import numpy as np
import faiss
from datetime import datetime

# ===================== 配置 =====================
DATA_PATH = "dji_chunks_final.jsonl"
CACHE_PATH = "embed_cache.npz"
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_metadata.json"
DIM = 384  # 嵌入维度


def load_data(max_chunks: int):
    """加载数据和嵌入"""
    print(f"[1/4] 加载数据 (最多 {max_chunks:,} 条)...")
    
    # 加载chunks
    chunks = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_chunks:
                break
            chunks.append(json.loads(line))
    
    # 加载嵌入
    cache = np.load(CACHE_PATH, allow_pickle=True)
    embeddings = cache['embeddings'][:len(chunks)].astype('float32')
    
    print(f"      加载了 {len(chunks):,} 条数据")
    print(f"      嵌入矩阵: {embeddings.shape}")
    return chunks, embeddings


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """
    构建FAISS索引
    
    Args:
        embeddings: 嵌入向量 [N, D]
        use_gpu: 是否使用GPU
    """
    print(f"\n[2/4] 构建FAISS索引 {'(GPU加速)' if use_gpu else '(CPU模式)'}...")
    
    n, d = embeddings.shape
    
    # 选择索引类型
    if n < 10000:
        # 小数据：暴力搜索
        print("      使用 IndexFlatIP (暴力搜索)")
        index = faiss.IndexFlatIP(d)  # 内积 = 余弦相似度（归一化后）
    else:
        # 大数据：IVF + PQ 量化
        # IVF: 倒排文件加速搜索
        # PQ: 乘积量化减少内存
        nlist = min(4096, int(np.sqrt(n)))  # 聚类中心数
        m = 32  # PQ子向量数
        
        print(f"      使用 IndexIVFPQ (nlist={nlist}, m={m})")
        print(f"      注意: IVF索引需要先训练")
        
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        
        # 训练索引（使用部分数据）
        train_size = min(100000, n)
        print(f"      用 {train_size:,} 条数据训练索引...")
        index.train(embeddings[:train_size])
    
    # 移动到GPU
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            # 设置GPU内存限制（防止OOM）
            res.setTempMemory(512 * 1024 * 1024)  # 512MB临时内存
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("      索引已移至GPU 0")
        except Exception as e:
            print(f"      GPU初始化失败: {e}")
            print("      回退到CPU模式")
    
    # 添加向量
    print(f"      添加 {n:,} 个向量到索引...")
    start = time.time()
    
    # 分批添加避免GPU内存溢出
    batch_size = 50000 if use_gpu else 10000
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        index.add(embeddings[i:end])
        if (i // batch_size + 1) % 5 == 0 or end == n:
            print(f"        已添加 {end:,}/{n:,}")
    
    elapsed = time.time() - start
    print(f"      索引构建完成，耗时 {elapsed:.1f} 秒")
    print(f"      速度: {n/elapsed:.0f} 向量/秒")
    
    # 如果是GPU索引，转回CPU保存
    if use_gpu and hasattr(index, 'device'):
        print("      将索引转回CPU以保存...")
        index = faiss.index_gpu_to_cpu(index)
    
    return index


def save_index(index: faiss.Index, chunks: list):
    """保存索引和元数据"""
    print(f"\n[3/4] 保存索引...")
    
    # 保存FAISS索引
    faiss.write_index(index, INDEX_PATH)
    print(f"      索引保存到: {INDEX_PATH}")
    
    # 保存元数据
    metadata = {
        'total_vectors': len(chunks),
        'created_at': datetime.now().isoformat(),
        'index_type': type(index).__name__,
        'embedding_dim': DIM,
        'items': []
    }
    
    for i, chunk in enumerate(chunks):
        metadata['items'].append({
            'id': f"chunk_{i}",
            'doc_id': chunk.get('doc_id', ''),
            'text': (chunk.get('text', '') or chunk.get('content', ''))[:200],
            'page': chunk.get('page', -1)
        })
    
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"      元数据保存到: {META_PATH}")


def validate(index: faiss.Index, embeddings: np.ndarray):
    """验证索引"""
    print(f"\n[4/4] 验证索引...")
    
    # 测试查询
    test_vec = embeddings[0:1]
    start = time.time()
    D, I = index.search(test_vec, 5)
    elapsed = (time.time() - start) * 1000
    
    print(f"      测试查询延迟: {elapsed:.2f} ms")
    print(f"      Top-5 结果: {I[0].tolist()}")
    print(f"      相似度: {D[0].tolist()}")
    
    # 如果是IVF索引，检查是否已训练
    if hasattr(index, 'ntotal'):
        print(f"      索引中向量数: {index.ntotal}")
    
    return {
        'latency_ms': round(elapsed, 2),
        'total_vectors': index.ntotal if hasattr(index, 'ntotal') else 'unknown'
    }


def search_demo(index_path: str, meta_path: str, query_idx: int = 0):
    """演示搜索功能"""
    print(f"\n[演示] 搜索测试...")
    
    # 加载
    index = faiss.read_index(index_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # 从缓存获取查询向量
    cache = np.load(CACHE_PATH, allow_pickle=True)
    query_vec = cache['embeddings'][query_idx:query_idx+1].astype('float32')
    
    # 搜索
    D, I = index.search(query_vec, 5)
    
    print(f"\n查询第 {query_idx} 条数据的相似结果:")
    for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
        item = metadata['items'][idx]
        print(f"  [{rank}] ID: {item['id']}, Score: {score:.4f}")
        print(f"       Text: {item['text'][:80]}...")


def main():
    # 解析参数
    max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    use_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("=" * 60)
    print("DJI 向量数据库构建 - FAISS GPU版")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  处理数量: {max_chunks:,}")
    print(f"  使用GPU: {'是' if use_gpu else '否'}")
    
    total_start = time.time()
    
    # 1. 加载数据
    chunks, embeddings = load_data(max_chunks)
    
    # 2. 构建索引
    index = build_faiss_index(embeddings, use_gpu=bool(use_gpu))
    
    # 3. 保存
    save_index(index, chunks)
    
    # 4. 验证
    stats = validate(index, embeddings)
    
    # 5. 演示搜索
    search_demo(INDEX_PATH, META_PATH)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)
    print(f"\n总耗时: {total_elapsed:.1f} 秒")
    print(f"平均速度: {len(chunks)/total_elapsed:.0f} 向量/秒")
    print(f"\n产出文件:")
    print(f"  - {INDEX_PATH}  索引文件 (~{os.path.getsize(INDEX_PATH)/1024/1024:.0f}MB)")
    print(f"  - {META_PATH}  元数据文件")
    print(f"\n查询延迟: {stats['latency_ms']} ms")


if __name__ == '__main__':
    main()
