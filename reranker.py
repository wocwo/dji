"""
重排序器模块 (Reranker)

功能：
1. 使用重排序模型（如BAAI/bge-reranker）对候选文档进行精排
2. 基于意图对匹配类型的Chunk进行分数加权
3. 设置最低分数阈值进行噪声过滤
4. 输出按分数降序排列的候选Chunk集
"""

import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from intent_classifier import PrimaryIntent, SecondaryIntent, IntentClassification
from multi_path_retriever import RecalledChunk, RetrievalChannel


@dataclass
class RerankedChunk:
    """重排序后的Chunk"""
    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    
    # 重排序分数
    rerank_score: float = 0.0
    
    # 原始RRF分数
    rrf_score: float = 0.0
    
    # 最终加权分数
    final_score: float = 0.0
    
    # 来源通道
    sources: List[RetrievalChannel] = field(default_factory=list)
    
    # 重排序模型原始输出
    raw_score: float = 0.0
    
    # 是否通过阈值过滤
    passed_threshold: bool = True


@dataclass
class RerankResult:
    """重排序结果"""
    query: str
    reranked_chunks: List[RerankedChunk]
    
    # 统计信息
    input_count: int
    output_count: int
    filtered_count: int
    avg_score: float
    max_score: float
    min_score: float
    rerank_time_ms: float
    
    # 阈值设置
    threshold: float


class BaseReranker:
    """重排序器基类"""
    
    def __init__(self, threshold: float = 0.3):
        """
        初始化重排序器
        
        Args:
            threshold: 最低分数阈值
        """
        self.threshold = threshold
    
    def rerank(
        self,
        query: str,
        chunks: List[RecalledChunk],
        intent: IntentClassification,
        top_k: int = 10
    ) -> RerankResult:
        """
        重排序
        
        Args:
            query: 查询文本
            chunks: 候选Chunk列表
            intent: 意图分类
            top_k: 输出Top K
            
        Returns:
            重排序结果
        """
        raise NotImplementedError
    
    def apply_intent_boost(
        self,
        chunk: RecalledChunk,
        score: float,
        intent: IntentClassification
    ) -> float:
        """
        基于意图应用分数加权
        
        Args:
            chunk: 候选Chunk
            score: 原始分数
            intent: 意图分类
            
        Returns:
            加权后的分数
        """
        boost = 1.0
        
        metadata = chunk.metadata
        chunk_type = metadata.get("type", "")
        chunk_section = metadata.get("section", "")
        
        # 根据意图类型匹配度调整权重
        intent_type_boosts = {
            PrimaryIntent.PARAMETER_QUERY: ["参数", "规格", "技术规格"],
            PrimaryIntent.OPERATION_QUERY: ["操作", "步骤", "指南"],
            PrimaryIntent.TROUBLESHOOTING: ["故障", "异常", "排查"],
            PrimaryIntent.FUNCTION_QUERY: ["功能", "特性"],
            PrimaryIntent.MAINTENANCE: ["保养", "维护"],
            PrimaryIntent.SAFETY_COMPLIANCE: ["安全", "法规"],
            PrimaryIntent.COMPARISON: ["对比", "参数"]
        }
        
        expected_types = intent_type_boosts.get(intent.primary_intent, [])
        
        if chunk_type in expected_types:
            boost += 0.15  # 类型匹配加15%
        
        if chunk_section in expected_types:
            boost += 0.1  # 章节匹配加10%
        
        # 二级意图精确匹配
        secondary_section_map = {
            SecondaryIntent.FLIGHT_PARAM: ["飞行参数", "飞行性能"],
            SecondaryIntent.BATTERY_PARAM: ["电池参数", "电源系统"],
            SecondaryIntent.TAKEOFF_LANDING: ["起飞降落"],
            SecondaryIntent.CALIBRATION: ["校准"],
            SecondaryIntent.LOST_DRONE: ["飞丢找回"],
            SecondaryIntent.ERROR_CODE: ["错误代码"],
        }
        
        expected_sections = secondary_section_map.get(intent.secondary_intent, [])
        if any(es in chunk_section for es in expected_sections):
            boost += 0.1
        
        return min(1.0, score * boost)


class CrossEncoderReranker(BaseReranker):
    """
    交叉编码器重排序器
    使用BAAI/bge-reranker系列模型
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        threshold: float = 0.3,
        batch_size: int = 8,
        device: str = "cpu"
    ):
        """
        初始化交叉编码器重排序器
        
        Args:
            model_name: 模型名称
            threshold: 分数阈值
            batch_size: 批大小
            device: 运行设备
        """
        super().__init__(threshold)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"[Reranker] 加载模型: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            print(f"[Reranker] 模型加载成功")
            
        except ImportError:
            print("[Reranker] 警告: sentence-transformers未安装，使用模拟模式")
            self.model = None
        except Exception as e:
            print(f"[Reranker] 模型加载失败: {e}，使用模拟模式")
            self.model = None
    
    def _compute_scores(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        计算查询-文档对的相关性分数
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            分数列表
        """
        if self.model is None:
            # 模拟模式：使用简单的关键词匹配
            return self._mock_scores(query, documents)
        
        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 批量预测
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )
        
        # 转换为0-1范围（bge-reranker输出通常是-10到10）
        scores = 1 / (1 + np.exp(-scores))  # sigmoid转换
        
        return scores.tolist()
    
    def _mock_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        模拟分数计算（用于测试）
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            模拟分数列表
        """
        scores = []
        query_words = set(query.lower().split())
        
        for doc in documents:
            # 简单的Jaccard相似度
            doc_words = set(doc.lower().split())
            if not query_words or not doc_words:
                scores.append(0.0)
                continue
            
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            score = intersection / union if union > 0 else 0.0
            
            # 增加一些随机性
            import random
            score = min(1.0, score + random.uniform(-0.1, 0.2))
            scores.append(max(0.0, score))
        
        return scores
    
    def rerank(
        self,
        query: str,
        chunks: List[RecalledChunk],
        intent: IntentClassification,
        top_k: int = 10
    ) -> RerankResult:
        """
        重排序
        
        Args:
            query: 查询文本
            chunks: 候选Chunk列表（通常是MultiPathRetriever的输出）
            intent: 意图分类
            top_k: 输出Top K
            
        Returns:
            重排序结果
        """
        import numpy as np
        start_time = time.time()
        
        input_count = len(chunks)
        
        if input_count == 0:
            return RerankResult(
                query=query,
                reranked_chunks=[],
                input_count=0,
                output_count=0,
                filtered_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                rerank_time_ms=0.0,
                threshold=self.threshold
            )
        
        # 提取文档内容
        documents = [chunk.document for chunk in chunks]
        
        # 计算重排序分数
        rerank_scores = self._compute_scores(query, documents)
        
        # 构建重排序结果
        reranked_chunks = []
        for chunk, rerank_score in zip(chunks, rerank_scores):
            # 应用意图加权
            final_score = self.apply_intent_boost(chunk, rerank_score, intent)
            
            # 判断是否通过阈值
            passed = final_score >= self.threshold
            
            rc = RerankedChunk(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                metadata=chunk.metadata,
                rerank_score=rerank_score,
                rrf_score=chunk.rrf_score,
                final_score=final_score,
                sources=chunk.sources,
                raw_score=rerank_score,
                passed_threshold=passed
            )
            reranked_chunks.append(rc)
        
        # 按最终分数排序
        reranked_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        # 过滤低分Chunk
        passed_chunks = [rc for rc in reranked_chunks if rc.passed_threshold]
        filtered_count = len(reranked_chunks) - len(passed_chunks)
        
        # 取Top K
        output_chunks = passed_chunks[:top_k]
        
        rerank_time = (time.time() - start_time) * 1000
        
        # 统计分数
        if output_chunks:
            scores = [rc.final_score for rc in output_chunks]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        return RerankResult(
            query=query,
            reranked_chunks=output_chunks,
            input_count=input_count,
            output_count=len(output_chunks),
            filtered_count=filtered_count,
            avg_score=avg_score,
            max_score=max_score,
            min_score=min_score,
            rerank_time_ms=rerank_time,
            threshold=self.threshold
        )


class SimpleReranker(BaseReranker):
    """
    简单重排序器（不使用模型，仅基于RRF分数和意图加权）
    适用于没有GPU或模型资源受限的场景
    """
    
    def __init__(self, threshold: float = 0.3):
        """
        初始化简单重排序器
        
        Args:
            threshold: 分数阈值
        """
        super().__init__(threshold)
    
    def rerank(
        self,
        query: str,
        chunks: List[RecalledChunk],
        intent: IntentClassification,
        top_k: int = 10
    ) -> RerankResult:
        """
        简单重排序
        
        Args:
            query: 查询文本
            chunks: 候选Chunk列表
            intent: 意图分类
            top_k: 输出Top K
            
        Returns:
            重排序结果
        """
        start_time = time.time()
        
        input_count = len(chunks)
        
        if input_count == 0:
            return RerankResult(
                query=query,
                reranked_chunks=[],
                input_count=0,
                output_count=0,
                filtered_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                rerank_time_ms=0.0,
                threshold=self.threshold
            )
        
        # 基于RRF分数和意图加权进行重排序
        reranked_chunks = []
        for chunk in chunks:
            # 归一化RRF分数到0-1
            rrf_score = min(1.0, chunk.rrf_score)
            
            # 应用意图加权
            final_score = self.apply_intent_boost(chunk, rrf_score, intent)
            
            # 判断是否通过阈值
            passed = final_score >= self.threshold
            
            rc = RerankedChunk(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                metadata=chunk.metadata,
                rerank_score=rrf_score,
                rrf_score=chunk.rrf_score,
                final_score=final_score,
                sources=chunk.sources,
                raw_score=rrf_score,
                passed_threshold=passed
            )
            reranked_chunks.append(rc)
        
        # 按最终分数排序
        reranked_chunks.sort(key=lambda x: x.final_score, reverse=True)
        
        # 过滤低分Chunk
        passed_chunks = [rc for rc in reranked_chunks if rc.passed_threshold]
        filtered_count = len(reranked_chunks) - len(passed_chunks)
        
        # 取Top K
        output_chunks = passed_chunks[:top_k]
        
        rerank_time = (time.time() - start_time) * 1000
        
        # 统计分数
        if output_chunks:
            scores = [rc.final_score for rc in output_chunks]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
        else:
            avg_score = max_score = min_score = 0.0
        
        return RerankResult(
            query=query,
            reranked_chunks=output_chunks,
            input_count=input_count,
            output_count=len(output_chunks),
            filtered_count=filtered_count,
            avg_score=avg_score,
            max_score=max_score,
            min_score=min_score,
            rerank_time_ms=rerank_time,
            threshold=self.threshold
        )


def create_reranker(
    model_type: str = "simple",
    threshold: float = 0.3,
    **kwargs
) -> BaseReranker:
    """
    创建重排序器工厂函数
    
    Args:
        model_type: 模型类型 ("simple", "cross_encoder")
        threshold: 分数阈值
        **kwargs: 额外参数
        
    Returns:
        重排序器实例
    """
    if model_type == "cross_encoder":
        return CrossEncoderReranker(
            threshold=threshold,
            model_name=kwargs.get("model_name", "BAAI/bge-reranker-base"),
            device=kwargs.get("device", "cpu"),
            batch_size=kwargs.get("batch_size", 8)
        )
    else:
        return SimpleReranker(threshold=threshold)


if __name__ == "__main__":
    print("=" * 60)
    print("重排序器模块")
    print("=" * 60)
    print("\n支持的重排序器:")
    print("1. SimpleReranker: 基于RRF分数和意图加权的轻量级重排序")
    print("2. CrossEncoderReranker: 使用BAAI/bge-reranker模型的精确重排序")
    print("\n使用方式:")
    print("from reranker import create_reranker")
    print("reranker = create_reranker('cross_encoder', threshold=0.3)")
    print("result = reranker.rerank(query, chunks, intent, top_k=10)")
