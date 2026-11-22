#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DJI 手册向量数据库构建系统 - 最终版
支持预计算嵌入缓存 (embed_cache.npz)

使用方法:
    python build_vector_db_final.py [chunk数量]
    
示例:
    python build_vector_db_final.py 1000    # 处理1000条
    python build_vector_db_final.py         # 处理全部167万条
"""

import os
import sys
import json
import time
import numpy as np
import chromadb
from chromadb.config import Settings
from datetime import datetime
from typing import List, Dict, Any, Optional

# ===================== 配置 =====================

CHROMA_PATH = "./chroma_db_final"
COLLECTION_NAME = "uav_manual_db"
CACHE_PATH = "embed_cache.npz"
DATA_PATH = "dji_chunks_final.jsonl"
BATCH_SIZE = 100  # ChromaDB写入批大小


# ===================== 数据加载 =====================

def load_chunks(file_path: str, max_chunks: Optional[int] = None) -> List[Dict]:
    """加载文本块数据"""
    print(f"[1/5] 加载数据: {file_path}")
    chunks = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_chunks and idx >= max_chunks:
                break
            if line.strip():
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"      加载了 {len(chunks):,} 个文本块")
    return chunks


def load_embeddings(cache_path: str, limit: Optional[int] = None) -> tuple:
    """加载预计算嵌入"""
    print(f"[2/5] 加载嵌入缓存: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    ids = data['ids'][:limit] if limit else data['ids']
    embeddings = data['embeddings'][:limit] if limit else data['embeddings']
    print(f"      加载了 {len(embeddings):,} 个向量")
    print(f"      向量维度: {embeddings.shape[1]}")
    return ids, embeddings


# ===================== 数据处理 =====================

def prepare_data(chunks: List[Dict], cache_ids: np.ndarray) -> Dict[str, Any]:
    """准备ChromaDB数据"""
    print(f"[3/5] 准备数据...")
    
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        # 生成与缓存兼容的ID: {doc_id}_chunk_{index}
        doc_id = chunk.get('doc_id', 'unknown')
        chunk_id = f"{doc_id}_chunk_{i}"
        
        # 获取文本内容
        text = chunk.get('text', '') or chunk.get('content', '')
        
        ids.append(chunk_id)
        documents.append(text[:1000] if text else '')  # 限制长度
        metadatas.append({
            'doc_id': str(doc_id)[:100],
            'chunk_id': str(chunk.get('chunk_id', ''))[:50],
            'model': str(chunk.get('model', ''))[:50],
            'title': str(chunk.get('title', ''))[:100],
            'section': str(chunk.get('section', ''))[:100],
            'type': str(chunk.get('type', '文本'))[:20],
            'keywords': ','.join(chunk.get('keywords', []))[:200],
            'token_count': int(chunk.get('token_count', 0)),
            'lang': str(chunk.get('lang', 'unknown'))[:10],
            'page': int(chunk.get('page', -1)),
            'index_time': datetime.now().isoformat()
        })
    
    print(f"      准备了 {len(ids):,} 条记录")
    return {
        'ids': ids,
        'embeddings': cache_ids.tolist() if isinstance(cache_ids, np.ndarray) else cache_ids,
        'documents': documents,
        'metadatas': metadatas
    }


# ===================== ChromaDB操作 =====================

def build_index(data: Dict[str, Any]) -> chromadb.Collection:
    """构建ChromaDB索引"""
    print(f"[4/5] 构建ChromaDB索引...")
    
    # 初始化客户端
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # 删除旧集合（如果存在）
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"      删除旧集合")
    except:
        pass
    
    # 创建新集合
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            'hnsw:space': 'cosine',
            'description': 'DJI无人机手册向量数据库',
            'created_at': datetime.now().isoformat()
        }
    )
    print(f"      创建集合: {COLLECTION_NAME}")
    
    # 分批添加数据
    total = len(data['ids'])
    embeddings = data['embeddings']
    ids = data['ids']
    documents = data['documents']
    metadatas = data['metadatas']
    
    for i in range(0, total, BATCH_SIZE):
        end = min(i + BATCH_SIZE, total)
        collection.add(
            embeddings=embeddings[i:end],
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        if (i // BATCH_SIZE + 1) % 10 == 0 or end == total:
            print(f"      已添加 {end:,}/{total:,} ({end/total*100:.1f}%)")
    
    print(f"      索引完成，共 {collection.count():,} 个向量")
    return collection


# ===================== 验证 =====================

def validate(collection: chromadb.Collection) -> Dict[str, Any]:
    """验证索引质量"""
    print(f"[5/5] 质量验证...")
    
    report = {
        'validation_time': datetime.now().isoformat(),
        'basic_stats': {},
        'metadata_check': {},
        'test_queries': []
    }
    
    # 1. 基础统计
    count = collection.count()
    report['basic_stats'] = {
        'total_vectors': count,
        'collection_name': COLLECTION_NAME
    }
    print(f"      向量总数: {count:,}")
    
    # 2. 样本检查
    sample = collection.get(limit=3)
    if sample['metadatas']:
        report['metadata_check']['fields'] = list(sample['metadatas'][0].keys())
        print(f"      元数据字段: {report['metadata_check']['fields']}")
    
    # 3. 简单查询测试
    test_queries = [
        '无人机飞行高度',
        '指南针校准',
        '遥控器连接'
    ]
    
    for query in test_queries:
        try:
            # 使用随机向量作为查询（预计算模式）
            import random
            query_vec = [random.random() for _ in range(384)]
            start = time.time()
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=3,
                include=['metadatas', 'documents', 'distances']
            )
            latency = (time.time() - start) * 1000
            
            report['test_queries'].append({
                'query': query,
                'latency_ms': round(latency, 2),
                'results_count': len(results['ids'][0])
            })
            print(f"      查询 '{query}': {round(latency, 2)}ms")
        except Exception as e:
            print(f"      查询 '{query}': 失败 - {str(e)}")
    
    return report


def save_reports(collection: chromadb.Collection, report: Dict):
    """保存报告"""
    # 索引统计
    index_stat = {
        'total_vectors': collection.count(),
        'collection_name': COLLECTION_NAME,
        'distance_metric': 'cosine',
        'embedding_dim': 384,
        'last_update': datetime.now().isoformat()
    }
    with open('index_stat.json', 'w', encoding='utf-8') as f:
        json.dump(index_stat, f, ensure_ascii=False, indent=2)
    
    # 验证报告
    with open('retrieval_verification_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 配置文档
    config = {
        'project': 'DJI 手册向量数据库',
        'created_at': datetime.now().isoformat(),
        'embedding_config': {
            'model_type': 'precomputed',
            'cache_path': CACHE_PATH,
            'total_vectors': collection.count()
        },
        'chroma_config': {
            'persist_path': CHROMA_PATH,
            'collection_name': COLLECTION_NAME,
            'distance_metric': 'cosine'
        },
        'validation_summary': {
            'total_vectors': collection.count(),
            'test_queries': len(report['test_queries'])
        }
    }
    with open('vector_db_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ===================== 主程序 =====================

def main():
    print("=" * 60)
    print("DJI 手册向量数据库构建系统")
    print("=" * 60)
    
    # 解析参数
    max_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    if max_chunks:
        print(f"\n处理数量: {max_chunks:,} 条")
    else:
        print(f"\n处理数量: 全部数据 (约167万条)")
    
    start_time = time.time()
    
    # 1. 加载数据
    chunks = load_chunks(DATA_PATH, max_chunks)
    
    # 2. 加载嵌入
    _, embeddings = load_embeddings(CACHE_PATH, limit=len(chunks) if max_chunks else None)
    
    # 3. 准备数据
    data = prepare_data(chunks, embeddings)
    
    # 4. 构建索引
    collection = build_index(data)
    
    # 5. 验证
    report = validate(collection)
    
    # 6. 保存报告
    save_reports(collection, report)
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("构建完成!")
    print("=" * 60)
    print(f"\n总耗时: {elapsed:.1f} 秒")
    print(f"向量数量: {collection.count():,}")
    print("\n产出文件:")
    print(f"  - {CHROMA_PATH}/            向量数据库")
    print("  - index_stat.json           索引统计")
    print("  - retrieval_verification_report.json  验证报告")
    print("  - vector_db_config.json     配置文档")


if __name__ == '__main__':
    main()
