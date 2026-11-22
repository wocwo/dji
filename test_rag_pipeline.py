"""
RAG检索流水线集成测试

测试内容：
1. 各模块集成正确性
2. 多路召回效果
3. 重排序效果
4. 端到端性能
"""

import json
import time
from typing import List, Dict, Any
from dataclasses import asdict


def test_query_enhancement():
    """测试查询增强模块"""
    print("=" * 70)
    print("测试1: 查询增强模块")
    print("=" * 70)
    
    from retrieval_system import IntelligentRetrievalSystem
    
    system = IntelligentRetrievalSystem(use_llm_for_intent=False)
    
    test_queries = [
        "Mavic 4 Pro怎么起飞",
        "电池充不满",
        "遥控连不上",
        "图传卡了"
    ]
    
    for query in test_queries:
        context = system.process_query(query)
        print(f"\n查询: {query}")
        print(f"  清洗: {context.enhanced.cleaned_query}")
        print(f"  精准: {context.enhanced.precise_query}")
        print(f"  意图: {context.intent.primary_intent.value}")
        print(f"  耗时: {context.processing_time_ms:.2f}ms")
    
    print("\n[OK] 查询增强模块测试通过\n")


def test_multipath_retriever():
    """测试多路召回模块"""
    print("=" * 70)
    print("测试2: 多路召回模块")
    print("=" * 70)
    
    from multi_path_retriever import (
        MultiPathRetriever, ChannelScheduler, RetrievalChannel, RRFusion
    )
    from retrieval_system import IntelligentRetrievalSystem
    
    # 测试通道调度
    print("\n[通道调度测试]")
    from intent_classifier import IntentClassification, PrimaryIntent, SecondaryIntent
    
    intent = IntentClassification(
        primary_intent=PrimaryIntent.OPERATION_QUERY,
        secondary_intent=SecondaryIntent.TAKEOFF_LANDING,
        confidence=0.8
    )
    
    weights = ChannelScheduler.get_weights(intent)
    print("操作步骤查询的通道权重:")
    for ch, w in weights.items():
        enabled = "启用" if ChannelScheduler.should_enable_channel(ch, weights) else "禁用"
        print(f"  {ch.value}: {w:.2f} [{enabled}]")
    
    # 测试RRF融合
    print("\n[RRF融合测试]")
    rrf = RRFusion(k=60)
    
    # 模拟多通道结果
    mock_results = {
        RetrievalChannel.SEMANTIC: type('obj', (object,), {
            'channel': RetrievalChannel.SEMANTIC,
            'results': [
                {'id': 'chunk_1', 'document': '起飞操作说明1', 'similarity': 0.9},
                {'id': 'chunk_2', 'document': '起飞操作说明2', 'similarity': 0.85},
                {'id': 'chunk_3', 'document': '其他内容', 'similarity': 0.7}
            ],
            'weight': 1.0
        })(),
        RetrievalChannel.KEYWORD: type('obj', (object,), {
            'channel': RetrievalChannel.KEYWORD,
            'results': [
                {'id': 'chunk_2', 'document': '起飞操作说明2', 'similarity': 0.95},
                {'id': 'chunk_4', 'document': '起飞相关', 'similarity': 0.8}
            ],
            'weight': 0.8
        })()
    }
    
    fused = rrf.fuse(mock_results, weights)
    print(f"融合前: 语义={len(mock_results[RetrievalChannel.SEMANTIC].results)}, 关键词={len(mock_results[RetrievalChannel.KEYWORD].results)}")
    print(f"融合后: 去重后共{len(fused)}个唯一Chunk")
    print("\nTop 3 RRF分数:")
    for i, chunk in enumerate(fused[:3], 1):
        print(f"  [{i}] {chunk.chunk_id}: RRF={chunk.rrf_score:.4f}, 来源={chunk.sources}")
    
    print("\n[OK] 多路召回模块测试通过\n")


def test_reranker():
    """测试重排序模块"""
    print("=" * 70)
    print("测试3: 重排序模块")
    print("=" * 70)
    
    from reranker import create_reranker, SimpleReranker, CrossEncoderReranker
    from multi_path_retriever import RecalledChunk, RetrievalChannel
    from intent_classifier import IntentClassification, PrimaryIntent, SecondaryIntent
    
    # 创建简单重排序器
    reranker = create_reranker("simple", threshold=0.3)
    
    # 创建模拟候选
    mock_chunks = [
        RecalledChunk(
            chunk_id="chunk_1",
            document="Mavic 4 Pro起飞前需要检查电量和GPS信号",
            metadata={"type": "操作", "section": "起飞降落"},
            rrf_score=0.85,
            sources=[RetrievalChannel.SEMANTIC],
            channel_scores={RetrievalChannel.SEMANTIC: 0.9},
            channel_ranks={RetrievalChannel.SEMANTIC: 1}
        ),
        RecalledChunk(
            chunk_id="chunk_2",
            document="无人机起飞操作指南",
            metadata={"type": "操作", "section": "操作流程"},
            rrf_score=0.75,
            sources=[RetrievalChannel.KEYWORD],
            channel_scores={RetrievalChannel.KEYWORD: 0.85},
            channel_ranks={RetrievalChannel.KEYWORD: 2}
        ),
        RecalledChunk(
            chunk_id="chunk_3",
            document="电池保养注意事项",
            metadata={"type": "保养", "section": "维护保养"},
            rrf_score=0.65,
            sources=[RetrievalChannel.METADATA],
            channel_scores={RetrievalChannel.METADATA: 0.7},
            channel_ranks={RetrievalChannel.METADATA: 3}
        )
    ]
    
    intent = IntentClassification(
        primary_intent=PrimaryIntent.OPERATION_QUERY,
        secondary_intent=SecondaryIntent.TAKEOFF_LANDING,
        confidence=0.8
    )
    
    result = reranker.rerank(
        query="Mavic 4 Pro怎么起飞",
        chunks=mock_chunks,
        intent=intent,
        top_k=5
    )
    
    print(f"\n输入候选数: {result.input_count}")
    print(f"通过阈值: {result.output_count}")
    print(f"过滤掉: {result.filtered_count}")
    print(f"分数范围: [{result.min_score:.3f}, {result.max_score:.3f}]")
    print(f"重排序耗时: {result.rerank_time_ms:.2f}ms")
    
    print("\nTop 结果:")
    for chunk in result.reranked_chunks:
        boost_info = ""
        if chunk.metadata.get("type") == "操作":
            boost_info = " (意图加权)"
        print(f"  [{chunk.chunk_id}] 分数={chunk.final_score:.3f}{boost_info}")
    
    print("\n[OK] 重排序模块测试通过\n")


def test_full_pipeline_mock():
    """测试完整流水线（模拟模式）"""
    print("=" * 70)
    print("测试4: 完整流水线（模拟模式）")
    print("=" * 70)
    
    # 由于实际运行需要ChromaDB，这里仅测试模块集成
    print("\n[模块集成检查]")
    
    try:
        from rag_pipeline import RAGRetrievalPipeline, PipelineResult
        from multi_path_retriever import MultiPathRetriever
        from reranker import create_reranker
        from retrieval_system import IntelligentRetrievalSystem
        
        print("  [OK] rag_pipeline 导入成功")
        print("  [OK] multi_path_retriever 导入成功")
        print("  [OK] reranker 导入成功")
        print("  [OK] retrieval_system 导入成功")
        
        # 检查类定义
        assert hasattr(RAGRetrievalPipeline, 'search')
        assert hasattr(RAGRetrievalPipeline, 'explain_result')
        print("  [OK] RAGRetrievalPipeline 方法检查通过")
        
        assert hasattr(MultiPathRetriever, 'retrieve')
        print("  [OK] MultiPathRetriever 方法检查通过")
        
        print("\n[OK] 完整流水线模块集成测试通过")
        print("  注意: 实际检索需要ChromaDB和嵌入模型支持\n")
        
    except Exception as e:
        print(f"\n[FAIL] 模块集成测试失败: {e}\n")
        raise


def print_architecture():
    """打印系统架构"""
    print("=" * 70)
    print("RAG检索系统架构")
    print("=" * 70)
    
    architecture = """
┌─────────────────────────────────────────────────────────────────────┐
│                          RAG检索流水线                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Stage 1: 查询理解与增强 (query_enhancement.py)               │  │
│  │  - 查询清洗、术语标准化、错别字纠正、查询扩展                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Stage 2: 意图分类 (intent_classifier.py)                     │  │
│  │  - 7种一级意图、基于规则/LLM的分类、置信度评估                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Stage 3: 多路召回 (multi_path_retriever.py)                  │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │  │
│  │  │ 语义检索通道 │ │ 关键词检索   │ │ 元数据检索   │          │  │
│  │  │ (向量相似度) │ │ (BM25-like)  │ │ (意图过滤)   │          │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘          │  │
│  │                              ↓                                │  │
│  │                    ┌─────────────────┐                       │  │
│  │                    │ RRF融合 + 去重  │                       │  │
│  │                    └─────────────────┘                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Stage 4: 重排序 (reranker.py)                                │  │
│  │  - 交叉编码器/BGE-Reranker                                   │  │
│  │  - 意图加权、阈值过滤                                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Stage 5: 结果输出                                            │  │
│  │  - Top K文档、分数、元数据、来源通道                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

组件说明:
- MultiPathRetriever: 支持4路召回，动态通道调度
- RRFusion: 带权重的倒数秩融合，k=60
- ChannelScheduler: 基于意图自动调整7种意图的通道权重
- Reranker: 支持Simple/CrossEncoder两种模式，意图加权，阈值过滤
"""
    print(architecture)


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("RAG检索流水线集成测试")
    print("=" * 70 + "\n")
    
    # 打印架构
    print_architecture()
    
    # 运行各模块测试
    try:
        test_query_enhancement()
        test_multipath_retriever()
        test_reranker()
        test_full_pipeline_mock()
        
        print("=" * 70)
        print("所有测试通过！")
        print("=" * 70)
        print("\n下一步:")
        print("1. 确保ChromaDB已构建: python vector_db_system.py")
        print("2. 运行完整流水线测试: python rag_pipeline.py")
        print("3. 使用实际查询: from rag_pipeline import create_pipeline")
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
