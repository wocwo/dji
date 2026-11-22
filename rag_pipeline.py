"""
完整RAG检索流水线

整合：
1. 查询理解与增强
2. 意图分类
3. 多路召回（MultiPathRetriever）
4. RRF融合
5. 重排序（Reranker）
6. 上下文组装与提示词生成（ContextBuilder & PromptEngine）
7. 结果输出
"""

import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# 导入各模块
from retrieval_system import IntelligentRetrievalSystem, RetrievalContext
from multi_path_retriever import MultiPathRetriever, MultiPathResult, RecalledChunk
from reranker import create_reranker, RerankResult, RerankedChunk
from intent_classifier import IntentClassification
from context_builder import PromptEngine, AssembledContext, build_prompt


@dataclass
class GenerationContext:
    """生成阶段上下文"""
    assembled_context: AssembledContext
    prompt: str
    prompt_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """流水线完整结果"""
    
    # 原始查询
    query: str
    
    # 检索上下文
    context: RetrievalContext
    
    # 多路召回结果
    multipath_result: MultiPathResult
    
    # 重排序结果
    rerank_result: RerankResult
    
    # 最终结果（Top K文档）
    final_documents: List[Dict[str, Any]]
    
    # 性能统计
    total_time_ms: float
    stage_times: Dict[str, float]
    
    # 生成上下文（新增：上下文组装 + Prompt）
    generation_context: Optional[GenerationContext] = None
    
    # 时间戳
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RAGRetrievalPipeline:
    """
    完整RAG检索流水线
    
    处理流程：
    Query -> Query Enhancement -> Intent Classification -> Multi-Path Retrieval -> RRF Fusion -> Reranking -> Context Assembly & Prompt Engineering -> Final Results
    """
    
    def __init__(
        self,
        chroma_db,
        embedding_model,
        use_llm_for_intent: bool = False,
        reranker_type: str = "simple",
        reranker_threshold: float = 0.3,
        multipath_top_k: int = 50,
        final_top_k: int = 10,
        **reranker_kwargs
    ):
        """
        初始化RAG检索流水线
        
        Args:
            chroma_db: ChromaDB管理器
            embedding_model: 嵌入模型管理器
            use_llm_for_intent: 是否使用LLM进行意图分类
            reranker_type: 重排序器类型 ("simple" | "cross_encoder")
            reranker_threshold: 重排序分数阈值
            multipath_top_k: 多路召回数量
            final_top_k: 最终输出数量
            **reranker_kwargs: 重排序器额外参数
        """
        # 阶段1: 查询理解与意图分类
        self.query_system = IntelligentRetrievalSystem(
            use_llm_for_intent=use_llm_for_intent
        )
        
        # 阶段2: 多路召回
        self.multipath_retriever = MultiPathRetriever(
            chroma_db=chroma_db,
            embedding_model=embedding_model,
            default_top_k=multipath_top_k
        )
        
        # 阶段3: 重排序
        self.reranker = create_reranker(
            model_type=reranker_type,
            threshold=reranker_threshold,
            **reranker_kwargs
        )
        
        self.final_top_k = final_top_k
        
        # 阶段4: 上下文组装与提示词生成
        self.prompt_engine = PromptEngine(
            score_threshold=reranker_threshold,
            max_context_chars=8000
        )
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "avg_total_time_ms": 0,
            "avg_retrieval_time_ms": 0,
            "avg_rerank_time_ms": 0
        }
    
    def search(self, query: str) -> PipelineResult:
        """
        执行完整检索流程
        
        Args:
            query: 用户查询
            
        Returns:
            流水线完整结果
        """
        start_time = time.time()
        stage_times = {}
        
        # ========== 阶段1: 查询理解与意图分类 ==========
        stage_start = time.time()
        context = self.query_system.process_query(query)
        stage_times["query_enhancement"] = (time.time() - stage_start) * 1000
        
        # ========== 阶段2: 多路召回 ==========
        stage_start = time.time()
        multipath_result = self.multipath_retriever.retrieve(
            query=query,
            enhanced_query=context.enhanced.precise_query,
            expanded_queries=context.enhanced.expanded_queries,
            intent=context.intent,
            filter_condition=context.filter_condition,
            top_k=self.multipath_retriever.default_top_k
        )
        stage_times["multipath_retrieval"] = (time.time() - stage_start) * 1000
        
        # ========== 阶段3: 重排序 ==========
        stage_start = time.time()
        rerank_result = self.reranker.rerank(
            query=context.enhanced.precise_query,
            chunks=multipath_result.fused_results,
            intent=context.intent,
            top_k=self.final_top_k
        )
        stage_times["reranking"] = (time.time() - stage_start) * 1000
        
        # ========== 阶段4: 上下文组装与提示词生成 ==========
        stage_start = time.time()
        target_model = context.enhanced.detected_models[0] if context.enhanced.detected_models else None
        prompt_result = self.prompt_engine.build_prompt(
            query=query,
            rerank_result=rerank_result,
            intent=context.intent,
            target_model=target_model
        )
        generation_context = GenerationContext(
            assembled_context=prompt_result["context"],
            prompt=prompt_result["prompt"],
            prompt_metadata=prompt_result["metadata"]
        )
        stage_times["context_assembly"] = (time.time() - stage_start) * 1000
        
        # ========== 阶段5: 结果格式化 ==========
        stage_start = time.time()
        final_documents = self._format_final_results(rerank_result)
        stage_times["formatting"] = (time.time() - stage_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        stage_times["total"] = total_time
        
        # 更新统计
        self._update_stats(total_time, stage_times)
        
        return PipelineResult(
            query=query,
            context=context,
            multipath_result=multipath_result,
            rerank_result=rerank_result,
            final_documents=final_documents,
            generation_context=generation_context,
            total_time_ms=total_time,
            stage_times=stage_times
        )
    
    def _format_final_results(self, rerank_result: RerankResult) -> List[Dict[str, Any]]:
        """
        格式化最终结果
        
        Args:
            rerank_result: 重排序结果
            
        Returns:
            格式化的文档列表
        """
        documents = []
        
        for i, chunk in enumerate(rerank_result.reranked_chunks, start=1):
            doc = {
                "rank": i,
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "metadata": chunk.metadata,
                "final_score": round(chunk.final_score, 4),
                "rerank_score": round(chunk.rerank_score, 4),
                "rrf_score": round(chunk.rrf_score, 4),
                "sources": [s.value for s in chunk.sources],
                "passed_threshold": chunk.passed_threshold
            }
            documents.append(doc)
        
        return documents
    
    def _update_stats(self, total_time: float, stage_times: Dict[str, float]):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        n = self.stats["total_queries"]
        
        # 移动平均
        self.stats["avg_total_time_ms"] = (
            self.stats["avg_total_time_ms"] * (n - 1) + total_time
        ) / n
        
        self.stats["avg_retrieval_time_ms"] = (
            self.stats["avg_retrieval_time_ms"] * (n - 1) + stage_times.get("multipath_retrieval", 0)
        ) / n
        
        self.stats["avg_rerank_time_ms"] = (
            self.stats["avg_rerank_time_ms"] * (n - 1) + stage_times.get("reranking", 0)
        ) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def explain_result(self, result: PipelineResult) -> str:
        """
        解释检索结果
        
        Args:
            result: 流水线结果
            
        Returns:
            格式化说明字符串
        """
        lines = [
            "=" * 70,
            "RAG检索流水线结果",
            "=" * 70,
            f"",
            f"【原始查询】{result.query}",
            f"",
            f"【查询增强】",
            f"  清洗后: {result.context.enhanced.cleaned_query}",
            f"  精准查询: {result.context.enhanced.precise_query}",
            f"  查询类型: {result.context.enhanced.query_type.value}",
            f"  扩展查询数: {len(result.context.enhanced.expanded_queries)}",
            f"",
            f"【意图分类】",
            f"  一级意图: {result.context.intent.primary_intent.value}",
            f"  二级意图: {result.context.intent.secondary_intent.value}",
            f"  置信度: {result.context.intent.confidence:.2f}",
            f"",
            f"【多路召回】",
            f"  激活通道数: {result.multipath_result.active_channels}/{result.multipath_result.total_channels}",
            f"  召回Chunk数: {result.multipath_result.total_unique_chunks}",
            f"  各通道结果:"
        ]
        
        for ch, ch_result in result.multipath_result.channel_results.items():
            lines.append(f"    - {ch.value}: {len(ch_result.results)}个结果, 权重={ch_result.weight:.2f}")
        
        lines.extend([
            f"",
            f"【重排序】",
            f"  输入候选数: {result.rerank_result.input_count}",
            f"  通过阈值: {result.rerank_result.output_count}",
            f"  过滤掉: {result.rerank_result.filtered_count}",
            f"  分数范围: [{result.rerank_result.min_score:.3f}, {result.rerank_result.max_score:.3f}]",
            f"  平均分: {result.rerank_result.avg_score:.3f}",
            f"",
            f"【最终结果】Top {len(result.final_documents)} 文档:"
        ])
        
        for doc in result.final_documents[:5]:
            preview = doc["document"][:80] + "..." if len(doc["document"]) > 80 else doc["document"]
            lines.append(f"  [{doc['rank']}] 分数={doc['final_score']:.3f} | {preview}")
        
        lines.extend([
            f"",
            f"【性能统计】",
            f"  总耗时: {result.total_time_ms:.2f}ms",
            f"  - 查询增强: {result.stage_times.get('query_enhancement', 0):.2f}ms",
            f"  - 多路召回: {result.stage_times.get('multipath_retrieval', 0):.2f}ms",
            f"  - 重排序: {result.stage_times.get('reranking', 0):.2f}ms",
            f"  - 上下文组装: {result.stage_times.get('context_assembly', 0):.2f}ms",
            f"  - 结果格式化: {result.stage_times.get('formatting', 0):.2f}ms",
            "=" * 70
        ])
        
        if result.generation_context:
            lines.extend([
                f"",
                f"【生成上下文】",
                f"  组装Chunk数: {result.generation_context.assembled_context.chunk_count}",
                f"  上下文长度: {result.generation_context.assembled_context.total_chars} 字符",
                f"  Prompt长度: {result.generation_context.prompt_metadata.get('prompt_chars', 0)} 字符",
                f"  组装耗时: {result.generation_context.assembled_context.assembly_time_ms:.2f}ms"
            ])
        
        return "\n".join(lines)


class PipelineEvaluator:
    """流水线评估器"""
    
    def __init__(self, pipeline: RAGRetrievalPipeline):
        self.pipeline = pipeline
        self.evaluation_results = []
    
    def evaluate_queries(
        self,
        queries: List[str],
        expected_chunk_ids: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """
        评估查询集合
        
        Args:
            queries: 查询列表
            expected_chunk_ids: 期望的Chunk ID列表（用于计算Recall）
            
        Returns:
            评估结果
        """
        results = []
        total_time = 0
        
        for i, query in enumerate(queries):
            print(f"\n评估 [{i+1}/{len(queries)}]: {query}")
            
            result = self.pipeline.search(query)
            results.append(result)
            total_time += result.total_time_ms
            
            # 简单统计
            print(f"  召回: {result.multipath_result.total_unique_chunks} chunks")
            print(f"  输出: {len(result.final_documents)} docs")
            print(f"  耗时: {result.total_time_ms:.2f}ms")
            if result.generation_context:
                print(f"  Prompt长度: {result.generation_context.prompt_metadata.get('prompt_chars', 0)} 字符")
        
        # 计算指标
        avg_time = total_time / len(queries) if queries else 0
        
        report = {
            "total_queries": len(queries),
            "avg_time_ms": avg_time,
            "total_time_ms": total_time,
            "queries": [
                {
                    "query": r.query,
                    "intent": r.context.intent.primary_intent.value,
                    "recall_count": r.multipath_result.total_unique_chunks,
                    "output_count": len(r.final_documents),
                    "time_ms": r.total_time_ms,
                    "prompt_chars": r.generation_context.prompt_metadata.get("prompt_chars", 0) if r.generation_context else 0
                }
                for r in results
            ]
        }
        
        self.evaluation_results.append(report)
        
        return report
    
    def save_report(self, path: str = "pipeline_evaluation.json"):
        """保存评估报告"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        print(f"评估报告已保存: {path}")


# ============== 便捷函数 ==============

def create_pipeline(
    chroma_db_path: str = "./chroma_final",
    embedding_model_type: str = "precomputed",
    embedding_cache_path: str = "embed_cache.npz",
    **kwargs
) -> RAGRetrievalPipeline:
    """
    便捷函数：创建RAG检索流水线
    
    Args:
        chroma_db_path: ChromaDB路径
        embedding_model_type: 嵌入模型类型
        embedding_cache_path: 嵌入缓存路径
        **kwargs: 其他参数
        
    Returns:
        RAG检索流水线实例
    """
    from vector_db_system import (
        ChromaDBManager, EmbeddingModelManager,
        ChromaConfig, EmbeddingConfig
    )
    
    # 初始化嵌入模型
    embed_config = EmbeddingConfig(
        model_type=embedding_model_type,
        precomputed_cache_path=embedding_cache_path
    )
    embedding_model = EmbeddingModelManager(embed_config)
    
    # 初始化ChromaDB
    chroma_config = ChromaConfig(
        persist_path=chroma_db_path,
        collection_name="uav_manual_db"
    )
    chroma_db = ChromaDBManager(chroma_config)
    chroma_db.create_or_get_collection()
    
    # 创建流水线
    return RAGRetrievalPipeline(
        chroma_db=chroma_db,
        embedding_model=embedding_model,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 70)
    print("RAG检索流水线模块")
    print("=" * 70)
    print("\n完整处理流程:")
    print("1. 查询理解与增强")
    print("2. 意图分类")
    print("3. 多路召回 (语义检索 + 关键词检索 + 元数据检索 + 扩展查询)")
    print("4. RRF融合")
    print("5. 重排序")
    print("6. 上下文组装与提示词生成")
    print("7. 结果输出")
    print("\n使用方式:")
    print("from rag_pipeline import create_pipeline")
    print("pipeline = create_pipeline()")
    print("result = pipeline.search('Mavic 4 Pro怎么起飞')")
    print("print(pipeline.explain_result(result))")
    print("\n获取Prompt:")
    print("print(result.generation_context.prompt)")
