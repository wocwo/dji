"""
智能检索系统主模块
整合查询增强、意图分类、过滤条件生成，提供统一的检索接口
"""

import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from query_enhancement import QueryEnhancer, EnhancedQuery, QueryType
from intent_classifier import (
    HybridIntentClassifier, IntentClassification,
    PrimaryIntent, SecondaryIntent
)
from filter_generator import MetadataFilterGenerator, FilterCondition


@dataclass
class RetrievalContext:
    """检索上下文"""
    # 原始输入
    original_query: str
    
    # 查询增强结果
    enhanced: EnhancedQuery
    
    # 意图分类结果
    intent: IntentClassification
    
    # 过滤条件
    filter_condition: FilterCondition
    
    # 检索参数
    top_k: int = 5
    
    # 处理时间
    processing_time_ms: float = 0.0
    
    # 时间戳
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class RetrievalResult:
    """检索结果"""
    # 检索到的文档列表
    documents: List[Dict[str, Any]]
    
    # 检索上下文
    context: RetrievalContext
    
    # 元数据
    total_results: int = 0
    retrieval_time_ms: float = 0.0


class IntelligentRetrievalSystem:
    """
    智能检索系统
    
    处理流程：
    1. 查询增强（清洗、标准化、扩展）
    2. 意图分类（确定查询场景）
    3. 过滤条件生成（元数据过滤）
    4. 向量检索 + 过滤（需要外部向量数据库支持）
    """
    
    def __init__(
        self,
        use_llm_for_intent: bool = False,
        llm_api_key: Optional[str] = None
    ):
        """
        初始化智能检索系统
        
        Args:
            use_llm_for_intent: 是否使用LLM进行意图分类
            llm_api_key: LLM API密钥
        """
        self.query_enhancer = QueryEnhancer()
        self.intent_classifier = HybridIntentClassifier(
            use_llm=use_llm_for_intent,
            api_key=llm_api_key
        )
        self.filter_generator = MetadataFilterGenerator()
        
        # 统计信息
        self.stats = {
            "total_queries": 0,
            "enhanced_queries": 0,
            "classified_queries": 0,
            "avg_processing_time_ms": 0
        }
    
    def process_query(self, query: str) -> RetrievalContext:
        """
        处理查询（不包含实际检索）
        
        Args:
            query: 用户原始查询
            
        Returns:
            RetrievalContext对象
        """
        start_time = time.time()
        
        # 1. 查询增强
        enhanced = self.query_enhancer.enhance(query)
        
        # 2. 意图分类
        intent = self.intent_classifier.classify(enhanced.precise_query)
        
        # 3. 生成过滤条件
        filter_condition = self.filter_generator.generate(
            query=enhanced.precise_query,
            intent=intent,
            detected_models=enhanced.detected_models
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # 更新统计
        self.stats["total_queries"] += 1
        if enhanced.confidence > 0.5:
            self.stats["enhanced_queries"] += 1
        if intent.confidence > 0.5:
            self.stats["classified_queries"] += 1
        
        # 更新平均处理时间
        n = self.stats["total_queries"]
        old_avg = self.stats["avg_processing_time_ms"]
        self.stats["avg_processing_time_ms"] = (old_avg * (n - 1) + processing_time) / n
        
        return RetrievalContext(
            original_query=query,
            enhanced=enhanced,
            intent=intent,
            filter_condition=filter_condition,
            processing_time_ms=processing_time
        )
    
    def get_search_queries(self, context: RetrievalContext) -> Dict[str, Any]:
        """
        获取用于检索的查询列表
        
        Args:
            context: 检索上下文
            
        Returns:
            检索配置字典
        """
        return {
            "precise_query": context.enhanced.precise_query,
            "expanded_queries": context.enhanced.expanded_queries,
            "filter": context.filter_condition.chroma_filter,
            "top_k": context.top_k,
            "priority": context.filter_condition.priority
        }
    
    def get_system_prompt_context(self, context: RetrievalContext) -> str:
        """
        获取用于LLM的系统提示词上下文
        
        Args:
            context: 检索上下文
            
        Returns:
            格式化的提示词上下文
        """
        return f"""用户查询分析结果：
- 原始查询：{context.original_query}
- 清洗后查询：{context.enhanced.cleaned_query}
- 精准查询：{context.enhanced.precise_query}
- 查询意图：{context.intent.primary_intent.value} - {context.intent.secondary_intent.value}
- 检测到的机型：{', '.join(context.enhanced.detected_models) if context.enhanced.detected_models else '未指定'}
- 置信度：{context.intent.confidence:.2f}
"""
    
    def explain_processing(self, context: RetrievalContext) -> str:
        """
        解释查询处理过程
        
        Args:
            context: 检索上下文
            
        Returns:
            处理过程说明
        """
        lines = [
            "=" * 60,
            "查询处理过程",
            "=" * 60,
            f"",
            f"【原始输入】",
            f"  {context.original_query}",
            f"",
            f"【查询增强】",
            f"  清洗结果: {context.enhanced.cleaned_query}",
            f"  查询类型: {context.enhanced.query_type.value}",
            f"  精准查询: {context.enhanced.precise_query}",
        ]
        
        if context.enhanced.expanded_queries:
            lines.append(f"  扩展查询: {', '.join(context.enhanced.expanded_queries[:3])}")
        
        if context.enhanced.detected_typos:
            lines.append(f"  错别字纠正: {context.enhanced.detected_typos}")
        
        if context.enhanced.detected_abbreviations:
            lines.append(f"  术语标准化: {context.enhanced.detected_abbreviations}")
        
        if context.enhanced.detected_models:
            lines.append(f"  检测到机型: {', '.join(context.enhanced.detected_models)}")
        
        lines.extend([
            f"",
            f"【意图分类】",
            f"  一级意图: {context.intent.primary_intent.value}",
            f"  二级意图: {context.intent.secondary_intent.value}",
            f"  置信度: {context.intent.confidence:.2f}",
        ])
        
        if context.intent.detected_keywords:
            lines.append(f"  关键词: {', '.join(context.intent.detected_keywords)}")
        
        lines.extend([
            f"",
            f"【过滤条件】",
            f"  逻辑表达式: {context.filter_condition.raw_expression}",
            f"  优先级: {context.filter_condition.priority}",
            f"  描述: {context.filter_condition.description}",
            f"  预期效果: {context.filter_condition.expected_effect}",
            f"",
            f"【处理时间】{context.processing_time_ms:.2f}ms",
            "=" * 60
        ])
        
        return '\n'.join(lines)
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        return self.stats.copy()
    
    def batch_process(self, queries: List[str]) -> List[RetrievalContext]:
        """
        批量处理查询
        
        Args:
            queries: 查询列表
            
        Returns:
            检索上下文列表
        """
        return [self.process_query(q) for q in queries]


class RetrievalEvaluator:
    """检索评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_intent_accuracy(
        self, 
        contexts: List[RetrievalContext],
        expected_intents: List[PrimaryIntent]
    ) -> Dict:
        """
        评估意图识别准确率
        
        Args:
            contexts: 检索上下文列表
            expected_intents: 期望的一级意图列表
            
        Returns:
            评估结果
        """
        if len(contexts) != len(expected_intents):
            raise ValueError("上下文列表和期望意图列表长度不匹配")
        
        correct = 0
        details = []
        
        for ctx, expected in zip(contexts, expected_intents):
            predicted = ctx.intent.primary_intent
            is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            details.append({
                "query": ctx.original_query,
                "predicted": predicted.value,
                "expected": expected.value,
                "correct": is_correct,
                "confidence": ctx.intent.confidence
            })
        
        total = len(contexts)
        accuracy = correct / total if total > 0 else 0
        
        return {
            "metric": "intent_accuracy",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "details": details
        }
    
    def evaluate_query_rewrite_accuracy(
        self,
        contexts: List[RetrievalContext],
        expected_rewrites: List[str]
    ) -> Dict:
        """
        评估查询改写准确率
        
        Args:
            contexts: 检索上下文列表
            expected_rewrites: 期望的改写结果列表
            
        Returns:
            评估结果
        """
        if len(contexts) != len(expected_rewrites):
            raise ValueError("上下文列表和期望改写列表长度不匹配")
        
        correct = 0
        details = []
        
        for ctx, expected in zip(contexts, expected_rewrites):
            # 检查精准查询是否包含期望的关键词
            predicted = ctx.enhanced.precise_query
            # 简单判断：如果期望的关键词都在改写结果中，则认为正确
            expected_keywords = set(expected.split())
            predicted_keywords = set(predicted.split())
            
            # 计算重合度
            if expected_keywords:
                overlap = len(expected_keywords & predicted_keywords) / len(expected_keywords)
                is_correct = overlap >= 0.7  # 70%重叠度认为正确
            else:
                is_correct = predicted == expected
            
            if is_correct:
                correct += 1
            
            details.append({
                "query": ctx.original_query,
                "predicted": predicted,
                "expected": expected,
                "correct": is_correct,
                "overlap": overlap if expected_keywords else 1.0
            })
        
        total = len(contexts)
        accuracy = correct / total if total > 0 else 0
        
        return {
            "metric": "rewrite_accuracy",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "details": details
        }
    
    def evaluate_filter_accuracy(
        self,
        contexts: List[RetrievalContext],
        expected_has_filter: List[bool]
    ) -> Dict:
        """
        评估过滤条件生成准确率
        
        Args:
            contexts: 检索上下文列表
            expected_has_filter: 期望是否有过滤条件
            
        Returns:
            评估结果
        """
        if len(contexts) != len(expected_has_filter):
            raise ValueError("上下文列表和期望过滤列表长度不匹配")
        
        correct = 0
        details = []
        
        for ctx, expected in zip(contexts, expected_has_filter):
            # 检查是否生成了过滤条件
            has_filter = ctx.filter_condition.chroma_filter is not None
            is_correct = has_filter == expected
            
            if is_correct:
                correct += 1
            
            details.append({
                "query": ctx.original_query,
                "has_filter": has_filter,
                "expected_filter": expected,
                "correct": is_correct,
                "filter_expr": ctx.filter_condition.raw_expression
            })
        
        total = len(contexts)
        accuracy = correct / total if total > 0 else 0
        
        return {
            "metric": "filter_accuracy",
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
            "details": details
        }
    
    def generate_report(self, output_path: str = "retrieval_system_report.json"):
        """生成评估报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"报告已保存: {output_path}")


# ============== 便捷函数 ==============

def process_user_query(query: str, explain: bool = True) -> Dict:
    """
    便捷的用户查询处理函数
    
    Args:
        query: 用户查询
        explain: 是否输出处理过程说明
        
    Returns:
        处理结果字典
    """
    system = IntelligentRetrievalSystem(use_llm_for_intent=False)
    context = system.process_query(query)
    
    if explain:
        print(system.explain_processing(context))
    
    return {
        "original": context.original_query,
        "cleaned": context.enhanced.cleaned_query,
        "precise": context.enhanced.precise_query,
        "expanded": context.enhanced.expanded_queries,
        "query_type": context.enhanced.query_type.value,
        "primary_intent": context.intent.primary_intent.value,
        "secondary_intent": context.intent.secondary_intent.value,
        "intent_confidence": context.intent.confidence,
        "detected_models": context.enhanced.detected_models,
        "filter_expression": context.filter_condition.raw_expression,
        "chroma_filter": context.filter_condition.chroma_filter,
        "processing_time_ms": context.processing_time_ms
    }


if __name__ == "__main__":
    # 测试
    test_queries = [
        "请问Mavic 4 Pro怎么起飞啊？",
        "遥控器连不上怎么办",
        "图传画面卡顿",
        "电池充不满",
        "咋校准指南针",
        "飞丢了怎么找回",
        "固件怎么升级",
        "Mavic 3和Mavic 4有什么区别",
    ]
    
    system = IntelligentRetrievalSystem(use_llm_for_intent=False)
    
    print("=" * 80)
    print("智能检索系统测试")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n\n处理查询: {query}")
        print("-" * 80)
        result = system.process_query(query)
        print(system.explain_processing(result))
    
    print("\n\n系统统计信息:")
    print(json.dumps(system.get_stats(), indent=2))
