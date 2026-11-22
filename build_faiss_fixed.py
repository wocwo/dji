#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI 向量数据库构建 - FAISS 修复版
支持GPU加速构建，CPU保存索引

安装:
    # 方法1: conda (推荐)
    conda install -c pytorch faiss-gpu
    
    # 方法2: pip (CPU版)
    pip install faiss-cpu

使用:
    python build_faiss_fixed.py [chunk数量] [use_gpu]
"""

import os
import sys
import json
import time
import shutil
import numpy as np
from datetime import datetime

# 尝试导入FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("错误: 未安装FAISS")
    print("请运行: conda install -c pytorch faiss-gpu")
    print("   或: pip install faiss-cpu")
    sys.exit(1)

# ===================== 配置 =====================
DATA_PATH = "dji_chunks_final.jsonl"
CACHE_PATH = "embed_cache.npz"
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_metadata.json"
DIM = 384


def has_gpu() -> bool:
    """检查是否有可用GPU"""
    try:
        # 检查CUDA是否可用
        import torch
        return torch.cuda.is_available()
    except:
        # 直接检查faiss GPU资源
        try:
            res = faiss.StandardGpuResources()
            return True
        except:
            return False


def load_data(max_chunks: int):
    """加载数据和嵌入"""
    print(f"[1/4] 加载数据 (最多 {max_chunks:,} 条)...")
    
    chunks = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_chunks:
                break
            chunks.append(json.loads(line))
    
    cache = np.load(CACHE_PATH, allow_pickle=True)
    embeddings = cache['embeddings'][:len(chunks)].astype('float32')
    
    # 归一化向量（使内积=余弦相似度）
    faiss.normalize_L2(embeddings)
    
    print(f"      加载了 {len(chunks):,} 条数据")
    print(f"      嵌入矩阵: {embeddings.shape}")
    print(f"      已归一化（支持余弦相似度）")
    return chunks, embeddings


def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """构建FAISS索引"""
    
    # 自动检测GPU
    gpu_available = has_gpu()
    if use_gpu and not gpu_available:
        print("      ⚠️ 未检测到GPU，使用CPU模式")
        use_gpu = False
    
    print(f"\n[2/4] 构建FAISS索引 {'(GPU加速)' if use_gpu else '(CPU模式)'}...")
    
    n, d = embeddings.shape
    
    # 选择索引类型
    if n < 50000:
        # 小数据：暴力搜索
        print("      索引类型: IndexFlatIP (精确搜索)")
        index = faiss.IndexFlatIP(d)
    else:
        # 大数据：IVF + PQ
        nlist = min(4096, int(4 * np.sqrt(n)))
        m = 32  # PQ子向量数
        nbits = 8
        
        print(f"      索引类型: IndexIVFPQ (近似搜索)")
        print(f"      参数: nlist={nlist}, m={m}, nbits={nbits}")
        
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        
        # 训练
        train_size = min(256 * 100, n)
        print(f"      训练索引 ({train_size:,} 条数据)...")
        index.train(embeddings[:train_size])
    
    # 构建GPU索引（如果启用）
    gpu_index = None
    if use_gpu:
        try:
            print("      初始化GPU资源...")
            res = faiss.StandardGpuResources()
            res.setTempMemory(256 * 1024 * 1024)
            
            # 复制到GPU
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            print("      索引已移至GPU")
            target_index = gpu_index
        except Exception as e:
            print(f"      GPU初始化失败: {e}")
            target_index = index
            use_gpu = False
    else:
        target_index = index
    
    # 添加向量
    print(f"      添加 {n:,} 个向量...")
    start = time.time()
    
    # 批量添加
    batch_size = 100000 if use_gpu else 50000
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        target_index.add(embeddings[i:end])
        if (i // batch_size + 1) % 5 == 0 or end == n:
            print(f"        已添加 {end:,}/{n:,}")
    
    elapsed = time.time() - start
    print(f"      构建完成，耗时 {elapsed:.1f} 秒 ({n/elapsed:.0f} 向量/秒)")
    
    # 如果是GPU索引，转回CPU
    if use_gpu and gpu_index is not None:
        print("      将索引转回CPU以保存...")
        index = faiss.index_gpu_to_cpu(gpu_index)
    
    return index


def save_index(index: faiss.Index, chunks: list):
    """保存索引和元数据"""
    print(f"\n[3/4] 保存索引...")
    
    # 保存FAISS索引
    faiss.write_index(index, INDEX_PATH)
    file_size = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    print(f"      索引保存到: {INDEX_PATH} ({file_size:.1f} MB)")
    
    # 保存元数据（分批避免内存问题）
    print(f"      保存元数据...")
    metadata = {
        'total_vectors': len(chunks),
        'created_at': datetime.now().isoformat(),
        'index_type': str(type(index)).replace("<class '", "").replace("'>", ""),
        'embedding_dim': DIM,
    }
    
    # 单独保存items（可能很大）
    items_file = META_PATH.replace('.json', '_items.jsonl')
    with open(items_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            item = {
                'id': f"chunk_{i}",
                'doc_id': chunk.get('doc_id', ''),
                'text': (chunk.get('text', '') or chunk.get('content', ''))[:300],
                'page': chunk.get('page', -1)
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    metadata['items_file'] = items_file
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"      元数据保存到: {META_PATH}")
    print(f"      条目数据: {items_file}")


def validate(index: faiss.Index, embeddings: np.ndarray):
    """验证索引"""
    print(f"\n[4/4] 验证索引...")
    
    # 测试查询
    test_idx = 0
    test_vec = embeddings[test_idx:test_idx+1]
    
    start = time.time()
    D, I = index.search(test_vec, 5)
    elapsed = (time.time() - start) * 1000
    
    print(f"      测试查询延迟: {elapsed:.2f} ms")
    print(f"      查询向量 ID: chunk_{test_idx}")
    print(f"      Top-5 结果 ID: {[f'chunk_{i}' for i in I[0]]}")
    print(f"      相似度分数: {D[0].tolist()}")
    
    # 检查召回率（自身是否在Top-1）
    if I[0][0] == test_idx:
        print(f"      ✅ 自相似性检查通过 (自身在Top-1)")
    else:
        print(f"      ⚠️ 自身在 Top-{list(I[0]).index(test_idx)+1 if test_idx in I[0] else 'N/A'}")
    
    return {
        'latency_ms': round(elapsed, 2),
        'total_vectors': index.ntotal
    }


def search_demo():
    """演示搜索"""
    print(f"\n[演示] 搜索测试...")
    
    # 加载
    index = faiss.read_index(INDEX_PATH)
    
    # 从缓存获取几个查询向量
    cache = np.load(CACHE_PATH, allow_pickle=True)
    query_indices = [0, 100, 1000]
    
    print(f"\n测试3个查询:")
    for idx in query_indices:
        query_vec = cache['embeddings'][idx:idx+1].astype('float32')
        faiss.normalize_L2(query_vec)
        
        D, I = index.search(query_vec, 3)
        print(f"\n  查询 chunk_{idx}:")
        for rank, (i, score) in enumerate(zip(I[0], D[0]), 1):
            print(f"    [{rank}] chunk_{i} (相似度: {score:.4f})")


def main():
    # 参数解析
    max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    use_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print("=" * 60)
    print("DJI 向量数据库构建 - FAISS GPU/CPU版")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  处理数量: {max_chunks:,}")
    print(f"  GPU加速: {'自动检测' if use_gpu else '禁用'}")
    
    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"\n错误: 数据文件不存在: {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(CACHE_PATH):
        print(f"\n错误: 嵌入缓存不存在: {CACHE_PATH}")
        sys.exit(1)
    
    total_start = time.time()
    
    # 1. 加载数据
    chunks, embeddings = load_data(max_chunks)
    
    # 2. 构建索引
    index = build_faiss_index(embeddings, use_gpu=bool(use_gpu))
    
    # 3. 保存
    save_index(index, chunks)
    
    # 4. 验证
    stats = validate(index, embeddings)
    
    # 5. 演示
    search_demo()
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✅ 构建完成!")
    print("=" * 60)
    print(f"\n总耗时: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"平均速度: {len(chunks)/total_elapsed:.0f} 向量/秒")
    print(f"\n产出文件:")
    print(f"  - {INDEX_PATH}  FAISS索引文件")
    print(f"  - {META_PATH}  元数据")
    print(f"  - faiss_metadata_items.jsonl  条目数据")
    print(f"\n查询延迟: {stats['latency_ms']} ms")


if __name__ == '__main__':
    main()
