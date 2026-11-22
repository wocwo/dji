"""
多路检索引擎 (MultiPathRetriever)

功能：
1. 多路召回：语义检索、关键词检索、元数据检索
2. 动态通道调度：基于意图自动启停对应通道
3. RRF融合：带权重的倒数秩融合算法
4. 去重合并：同一Chunk保留最高RRF分数
"""

import re
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from intent_classifier import PrimaryIntent, SecondaryIntent, IntentClassification


class RetrievalChannel(Enum):
    """检索通道类型"""
    SEMANTIC = "semantic"           # 主路：精准语义检索（向量相似度）
    KEYWORD = "keyword"             # 辅路1：实体关键词精准检索
    METADATA = "metadata"           # 辅路2：元数据定向检索
    EXPANDED = "expanded"           # 辅路3：扩展查询召回


@dataclass
class ChannelResult:
    """单个通道的检索结果"""
    channel: RetrievalChannel
    results: List[Dict[str, Any]]   # 检索结果列表
    weight: float = 1.0             # 通道权重
    latency_ms: float = 0.0         # 检索耗时


@dataclass
class RecalledChunk:
    """召回的Chunk"""
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    
    # 各通道的原始分数
    channel_scores: Dict[RetrievalChannel, float] = field(default_factory=dict)
    
    # RRF融合后的分数
    rrf_score: float = 0.0
    
    # 来源通道
    sources: List[RetrievalChannel] = field(default_factory=list)
    
    # 各通道排名
    channel_ranks: Dict[RetrievalChannel, int] = field(default_factory=dict)


@dataclass
class MultiPathResult:
    """多路检索最终结果"""
    query: str
    enhanced_query: str
    
    # 各通道原始结果
    channel_results: Dict[RetrievalChannel, ChannelResult]
    
    # RRF融合后的结果
    fused_results: List[RecalledChunk]
    
    # 统计信息
    total_channels: int
    active_channels: int
    total_unique_chunks: int
    fusion_time_ms: float


class RRFusion:
    """
    倒数秩融合 (Reciprocal Rank Fusion)
    
    公式：RRF_score(d) = Σ(weight_k / (k + rank_d_k))
    其中：
    - weight_k: 通道k的权重
    - rank_d_k: 文档d在通道k中的排名
    - k: 常数（通常取60）
    """
    
    def __init__(self, k: int = 60):
        """
        初始化RRF
        
        Args:
            k: 排名常数，防止低排名文档分数过高
        """
        self.k = k
    
    def fuse(
        self,
        channel_results: Dict[RetrievalChannel, ChannelResult],
        weights: Optional[Dict[RetrievalChannel, float]] = None
    ) -> List[RecalledChunk]:
        """
        融合多通道结果
        
        Args:
            channel_results: 各通道检索结果
            weights: 各通道权重，默认1.0
            
        Returns:
            按RRF分数排序的Chunk列表
        """
        if weights is None:
            weights = {ch: 1.0 for ch in channel_results.keys()}
        
        # 收集所有Chunk的分数
        chunk_data: Dict[str, RecalledChunk] = {}
        
        for channel, ch_result in channel_results.items():
            weight = weights.get(channel, 1.0)
            
            for rank, result in enumerate(ch_result.results, start=1):
                chunk_id = result.get("id", result.get("chunk_id", ""))
                
                if not chunk_id:
                    continue
                
                # 创建或获取Chunk
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = RecalledChunk(
                        chunk_id=chunk_id,
                        document=result.get("document", result.get("text", "")),
                        metadata=result.get("metadata", {}),
                        channel_scores={},
                        channel_ranks={},
                        sources=[]
                    )
                
                chunk = chunk_data[chunk_id]
                
                # 计算RRF分数贡献
                rrf_contribution = weight / (self.k + rank)
                chunk.rrf_score += rrf_contribution
                
                # 记录来源
                chunk.sources.append(channel)
                chunk.channel_ranks[channel] = rank
                
                # 记录原始分数
                original_score = result.get("similarity", result.get("score", 0.0))
                chunk.channel_scores[channel] = original_score
        
        # 转换为列表并排序
        fused_results = list(chunk_data.values())
        fused_results.sort(key=lambda x: x.rrf_score, reverse=True)
        
        return fused_results


class ChannelScheduler:
    """
    通道调度器
    基于意图自动启停对应通道
    """
    
    # 意图到通道权重的映射
    INTENT_CHANNEL_WEIGHTS = {
        PrimaryIntent.PARAMETER_QUERY: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.8,
            RetrievalChannel.METADATA: 0.9,
            RetrievalChannel.EXPANDED: 0.6
        },
        PrimaryIntent.OPERATION_QUERY: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.9,
            RetrievalChannel.METADATA: 0.7,
            RetrievalChannel.EXPANDED: 0.5
        },
        PrimaryIntent.TROUBLESHOOTING: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.7,
            RetrievalChannel.METADATA: 0.5,
            RetrievalChannel.EXPANDED: 0.8  # 故障排查需要更多召回
        },
        PrimaryIntent.FUNCTION_QUERY: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.8,
            RetrievalChannel.METADATA: 0.6,
            RetrievalChannel.EXPANDED: 0.5
        },
        PrimaryIntent.MAINTENANCE: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.7,
            RetrievalChannel.METADATA: 0.8,
            RetrievalChannel.EXPANDED: 0.5
        },
        PrimaryIntent.SAFETY_COMPLIANCE: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.8,
            RetrievalChannel.METADATA: 0.7,
            RetrievalChannel.EXPANDED: 0.5
        },
        PrimaryIntent.COMPARISON: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.9,
            RetrievalChannel.METADATA: 0.6,
            RetrievalChannel.EXPANDED: 0.7
        },
        PrimaryIntent.UNKNOWN: {
            RetrievalChannel.SEMANTIC: 1.0,
            RetrievalChannel.KEYWORD: 0.5,
            RetrievalChannel.METADATA: 0.3,
            RetrievalChannel.EXPANDED: 0.5
        }
    }
    
    # 二级意图特殊权重调整
    SECONDARY_BOOST = {
        SecondaryIntent.LOST_DRONE: {RetrievalChannel.EXPANDED: 0.2},
        SecondaryIntent.FAULT_DIAGNOSIS: {RetrievalChannel.EXPANDED: 0.2},
        SecondaryIntent.ERROR_CODE: {RetrievalChannel.KEYWORD: 0.2},
    }
    
    @classmethod
    def get_weights(
        cls,
        intent: IntentClassification
    ) -> Dict[RetrievalChannel, float]:
        """
        获取通道权重
        
        Args:
            intent: 意图分类结果
            
        Returns:
            通道权重字典
        """
        base_weights = cls.INTENT_CHANNEL_WEIGHTS.get(
            intent.primary_intent,
            cls.INTENT_CHANNEL_WEIGHTS[PrimaryIntent.UNKNOWN]
        ).copy()
        
        # 应用二级意图权重调整
        if intent.secondary_intent in cls.SECONDARY_BOOST:
            boost = cls.SECONDARY_BOOST[intent.secondary_intent]
            for ch, delta in boost.items():
                if ch in base_weights:
                    base_weights[ch] = min(1.0, base_weights[ch] + delta)
        
        # 根据置信度调整
        if intent.confidence < 0.5:
            # 低置信度时，降低元数据通道权重，增加语义通道权重
            base_weights[RetrievalChannel.METADATA] *= 0.5
            base_weights[RetrievalChannel.SEMANTIC] = min(1.0, base_weights[RetrievalChannel.SEMANTIC] + 0.1)
        
        return base_weights
    
    @classmethod
    def should_enable_channel(
        cls,
        channel: RetrievalChannel,
        weights: Dict[RetrievalChannel, float],
        min_weight: float = 0.3
    ) -> bool:
        """
        判断是否启用通道
        
        Args:
            channel: 通道类型
            weights: 权重字典
            min_weight: 最小启用权重
            
        Returns:
            是否启用
        """
        return weights.get(channel, 0.0) >= min_weight


class KeywordRetriever:
    """
    关键词检索器
    基于实体关键词的BM25-like检索
    """
    
    def __init__(self):
        self.stopwords = set([
            "的", "了", "是", "在", "和", "与", "或", "有", "我", "你", "他",
            "这", "那", "什么", "怎么", "如何", "为什么", "哪个", "哪些",
            "请", "吗", "呢", "吧", "啊", "呀", "请问", "麻烦", "谢谢"
        ])
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询关键词
        
        Args:
            query: 查询文本
            
        Returns:
            关键词列表
        """
        # 分词（简单实现，可以使用jieba等分词库）
        # 这里使用字符级和短语级混合
        keywords = []
        
        # 2-4字符的短语
        for length in range(4, 1, -1):
            for i in range(len(query) - length + 1):
                phrase = query[i:i+length]
                if not all(c in self.stopwords for c in phrase):
                    keywords.append(phrase)
        
        # 去重并保持顺序
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen and len(kw) > 1:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # 最多10个关键词
    
    def score_document(
        self,
        query_keywords: List[str],
        document: str,
        metadata: Dict[str, Any]
    ) -> float:
        """
        计算文档与查询的相关性分数
        
        Args:
            query_keywords: 查询关键词
            document: 文档内容
            metadata: 文档元数据
            
        Returns:
            相关性分数
        """
        if not query_keywords:
            return 0.0
        
        doc_lower = document.lower()
        score = 0.0
        
        # 关键词匹配分数
        for i, kw in enumerate(query_keywords):
            # 标题权重更高
            weight = 1.0 / (i + 1)  # 位置越靠前权重越高
            
            # 在文档中的匹配
            count = doc_lower.count(kw.lower())
            score += count * weight * 0.5
            
            # 在元数据中的匹配（权重更高）
            title = metadata.get("title", "")
            section = metadata.get("section", "")
            keywords_meta = metadata.get("keywords", "")
            
            if kw.lower() in title.lower():
                score += weight * 2.0
            if kw.lower() in section.lower():
                score += weight * 1.5
            if kw.lower() in str(keywords_meta).lower():
                score += weight * 1.0
        
        # 归一化
        return min(1.0, score / len(query_keywords))


class MultiPathRetriever:
    """
    多路检索引擎
    
    整合多种检索方式，提供统一的多路召回接口
    """
    
    def __init__(
        self,
        chroma_db,  # ChromaDBManager实例
        embedding_model,  # EmbeddingModelManager实例
        rrf_k: int = 60,
        default_top_k: int = 20
    ):
        """
        初始化多路检索引擎
        
        Args:
            chroma_db: ChromaDB管理器
            embedding_model: 嵌入模型管理器
            rrf_k: RRF常数
            default_top_k: 默认每路召回数量
        """
        self.chroma_db = chroma_db
        self.embedding_model = embedding_model
        self.rrf = RRFusion(k=rrf_k)
        self.keyword_retriever = KeywordRetriever()
        self.default_top_k = default_top_k
    
    def retrieve_semantic(
        self,
        query: str,
        where_filter: Optional[Dict] = None,
        top_k: int = 20
    ) -> ChannelResult:
        """
        语义检索通道
        
        Args:
            query: 查询文本
            where_filter: 元数据过滤条件
            top_k: 召回数量
            
        Returns:
            通道结果
        """
        import time
        start = time.time()
        
        results = self.chroma_db.query(
            query_text=query,
            embedding_model=self.embedding_model,
            n_results=top_k,
            where_filter=where_filter
        )
        
        latency = (time.time() - start) * 1000
        
        return ChannelResult(
            channel=RetrievalChannel.SEMANTIC,
            results=results.get("results", []),
            latency_ms=latency
        )
    
    def retrieve_keyword(
        self,
        query: str,
        top_k: int = 20
    ) -> ChannelResult:
        """
        关键词检索通道
        
        Args:
            query: 查询文本
            top_k: 召回数量
            
        Returns:
            通道结果
        """
        import time
        start = time.time()
        
        # 提取关键词
        keywords = self.keyword_retriever.extract_keywords(query)
        
        # 从ChromaDB获取候选文档
        # 这里简化处理，实际可以从全文索引获取
        # 临时方案：使用关键词过滤的语义检索
        
        # 获取较多样本进行关键词打分
        all_results = self.chroma_db.query(
            query_text=query,
            embedding_model=self.embedding_model,
            n_results=100,  # 获取更多候选
            where_filter=None
        )
        
        # 关键词打分并重排序
        scored_results = []
        for result in all_results.get("results", []):
            doc = result.get("document", "")
            meta = result.get("metadata", {})
            score = self.keyword_retriever.score_document(keywords, doc, meta)
            
            if score > 0:
                result["score"] = score
                result["similarity"] = score
                scored_results.append(result)
        
        # 按分数排序
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        
        latency = (time.time() - start) * 1000
        
        return ChannelResult(
            channel=RetrievalChannel.KEYWORD,
            results=scored_results[:top_k],
            latency_ms=latency
        )
    
    def retrieve_metadata(
        self,
        query: str,
        intent: IntentClassification,
        top_k: int = 20
    ) -> ChannelResult:
        """
        元数据定向检索通道
        
        Args:
            query: 查询文本
            intent: 意图分类
            top_k: 召回数量
            
        Returns:
            通道结果
        """
        import time
        start = time.time()
        
        # 基于意图构建元数据过滤
        from filter_generator import MetadataFilterGenerator
        filter_gen = MetadataFilterGenerator()
        
        # 需要detected_models
        from query_enhancement import QueryEnhancer
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance(query)
        
        filter_condition = filter_gen.generate(
            query=query,
            intent=intent,
            detected_models=enhanced.detected_models
        )
        
        # 使用元数据过滤进行语义检索
        results = self.chroma_db.query(
            query_text=query,
            embedding_model=self.embedding_model,
            n_results=top_k,
            where_filter=filter_condition.chroma_filter
        )
        
        latency = (time.time() - start) * 1000
        
        return ChannelResult(
            channel=RetrievalChannel.METADATA,
            results=results.get("results", []),
            latency_ms=latency
        )
    
    def retrieve_expanded(
        self,
        expanded_queries: List[str],
        top_k: int = 10
    ) -> ChannelResult:
        """
        扩展查询检索通道
        
        Args:
            expanded_queries: 扩展查询列表
            top_k: 每路召回数量
            
        Returns:
            通道结果
        """
        import time
        start = time.time()
        
        all_results = []
        
        for eq in expanded_queries[:3]:  # 最多3个扩展查询
            results = self.chroma_db.query(
                query_text=eq,
                embedding_model=self.embedding_model,
                n_results=top_k
            )
            all_results.extend(results.get("results", []))
        
        # 去重（按ID）
        seen_ids = set()
        unique_results = []
        for r in all_results:
            rid = r.get("id", "")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                unique_results.append(r)
        
        latency = (time.time() - start) * 1000
        
        return ChannelResult(
            channel=RetrievalChannel.EXPANDED,
            results=unique_results[:top_k],
            latency_ms=latency
        )
    
    def retrieve(
        self,
        query: str,
        enhanced_query: str,
        expanded_queries: List[str],
        intent: IntentClassification,
        filter_condition=None,
        top_k: int = 50
    ) -> MultiPathResult:
        """
        执行多路检索
        
        Args:
            query: 原始查询
            enhanced_query: 增强后的精准查询
            expanded_queries: 扩展查询列表
            intent: 意图分类
            filter_condition: 过滤条件（可选）
            top_k: 每路召回数量
            
        Returns:
            多路检索结果
        """
        import time
        start = time.time()
        
        # 获取通道权重
        weights = ChannelScheduler.get_weights(intent)
        
        # 执行各通道检索
        channel_results = {}
        active_channels = 0
        
        # 1. 语义检索通道（始终启用）
        if ChannelScheduler.should_enable_channel(
            RetrievalChannel.SEMANTIC, weights
        ):
            ch_result = self.retrieve_semantic(
                enhanced_query,
                filter_condition.chroma_filter if filter_condition else None,
                top_k=top_k
            )
            ch_result.weight = weights[RetrievalChannel.SEMANTIC]
            channel_results[RetrievalChannel.SEMANTIC] = ch_result
            active_channels += 1
        
        # 2. 关键词检索通道
        if ChannelScheduler.should_enable_channel(
            RetrievalChannel.KEYWORD, weights
        ):
            ch_result = self.retrieve_keyword(enhanced_query, top_k=top_k)
            ch_result.weight = weights[RetrievalChannel.KEYWORD]
            channel_results[RetrievalChannel.KEYWORD] = ch_result
            active_channels += 1
        
        # 3. 元数据检索通道
        if ChannelScheduler.should_enable_channel(
            RetrievalChannel.METADATA, weights
        ):
            ch_result = self.retrieve_metadata(
                enhanced_query, intent, top_k=top_k
            )
            ch_result.weight = weights[RetrievalChannel.METADATA]
            channel_results[RetrievalChannel.METADATA] = ch_result
            active_channels += 1
        
        # 4. 扩展查询通道
        if ChannelScheduler.should_enable_channel(
            RetrievalChannel.EXPANDED, weights
        ) and len(expanded_queries) > 1:
            ch_result = self.retrieve_expanded(expanded_queries, top_k=top_k//2)
            ch_result.weight = weights[RetrievalChannel.EXPANDED]
            channel_results[RetrievalChannel.EXPANDED] = ch_result
            active_channels += 1
        
        # RRF融合
        fused_results = self.rrf.fuse(channel_results, weights)
        
        fusion_time = (time.time() - start) * 1000
        
        return MultiPathResult(
            query=query,
            enhanced_query=enhanced_query,
            channel_results=channel_results,
            fused_results=fused_results,
            total_channels=len(channel_results),
            active_channels=active_channels,
            total_unique_chunks=len(fused_results),
            fusion_time_ms=fusion_time
        )


# ============== 便捷函数 ==============

def retrieve_multi_path(
    query: str,
    chroma_db,
    embedding_model,
    enhanced_query: Optional[str] = None,
    expanded_queries: Optional[List[str]] = None,
    intent: Optional[IntentClassification] = None
) -> MultiPathResult:
    """
    便捷的多路检索函数
    
    Args:
        query: 用户查询
        chroma_db: ChromaDB管理器
        embedding_model: 嵌入模型管理器
        enhanced_query: 增强查询（可选）
        expanded_queries: 扩展查询（可选）
        intent: 意图分类（可选）
        
    Returns:
        多路检索结果
    """
    # 如果没有预处理结果，自动处理
    if enhanced_query is None or intent is None:
        from retrieval_system import IntelligentRetrievalSystem
        system = IntelligentRetrievalSystem()
        context = system.process_query(query)
        enhanced_query = context.enhanced.precise_query
        expanded_queries = context.enhanced.expanded_queries
        intent = context.intent
    
    retriever = MultiPathRetriever(chroma_db, embedding_model)
    
    return retriever.retrieve(
        query=query,
        enhanced_query=enhanced_query,
        expanded_queries=expanded_queries or [enhanced_query],
        intent=intent
    )


if __name__ == "__main__":
    print("=" * 60)
    print("多路检索引擎模块")
    print("=" * 60)
    print("\n核心功能:")
    print("1. 多路召回: 语义检索、关键词检索、元数据检索、扩展查询")
    print("2. 动态通道调度: 基于意图自动调整通道权重")
    print("3. RRF融合: 带权重的倒数秩融合算法")
    print("4. 去重合并: 同一Chunk保留最高RRF分数")
    print("\n使用方式:")
    print("from multi_path_retriever import MultiPathRetriever")
    print("retriever = MultiPathRetriever(chroma_db, embedding_model)")
    print("result = retriever.retrieve(...)")
