"""
元数据过滤条件生成模块
功能：
1. 基于查询意图和关键词生成元数据过滤条件
2. 支持AND/OR逻辑组合
3. 生成ChromaDB兼容的过滤语法
"""

import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from intent_classifier import (
    PrimaryIntent, SecondaryIntent, IntentClassification
)


class FilterOperator(Enum):
    """过滤操作符"""
    EQ = "$eq"           # 等于
    NE = "$ne"           # 不等于
    GT = "$gt"           # 大于
    GTE = "$gte"         # 大于等于
    LT = "$lt"           # 小于
    LTE = "$lte"         # 小于等于
    IN = "$in"           # 在列表中
    NIN = "$nin"         # 不在列表中
    AND = "$and"         # 逻辑与
    OR = "$or"           # 逻辑或
    CONTAINS = "$contains"  # 包含（用于文本）


@dataclass
class FilterCondition:
    """过滤条件"""
    # 原始逻辑表达式
    raw_expression: str
    
    # ChromaDB兼容语法
    chroma_filter: Optional[Dict[str, Any]] = None
    
    # 过滤优先级（数字越小优先级越高）
    priority: int = 1
    
    # 条件描述
    description: str = ""
    
    # 预期效果
    expected_effect: str = ""


class MetadataFilterGenerator:
    """元数据过滤条件生成器"""
    
    def __init__(self):
        self._init_intent_filter_mapping()
        self._init_model_keywords()
    
    def _init_intent_filter_mapping(self):
        """初始化意图到过滤条件的映射"""
        
        self.intent_filters = {
            # 参数查询 -> 过滤type为参数或规格
            PrimaryIntent.PARAMETER_QUERY: {
                "type": ["参数", "规格", "技术规格", "规格参数"],
                "section": ["规格参数", "技术参数", "参数说明"]
            },
            
            # 操作步骤查询 -> 过滤type为操作
            PrimaryIntent.OPERATION_QUERY: {
                "type": ["操作", "步骤", "操作指南", "操作说明"],
                "section": ["操作步骤", "操作流程", "使用说明"]
            },
            
            # 故障排查查询 -> 过滤type为故障
            PrimaryIntent.TROUBLESHOOTING: {
                "type": ["故障", "异常", "排查", "解决方案"],
                "section": ["故障排查", "异常处理", "常见问题"]
            },
            
            # 功能说明查询 -> 过滤type为功能
            PrimaryIntent.FUNCTION_QUERY: {
                "type": ["功能", "特性", "智能功能"],
                "section": ["功能说明", "功能介绍", "智能功能"]
            },
            
            # 维护保养查询 -> 过滤type为维护
            PrimaryIntent.MAINTENANCE: {
                "type": ["保养", "维护", "存储"],
                "section": ["维护保养", "存储指南", "日常维护"]
            },
            
            # 安全合规查询 -> 过滤type为安全
            PrimaryIntent.SAFETY_COMPLIANCE: {
                "type": ["安全", "合规", "法规"],
                "section": ["安全须知", "法规要求", "合规说明"]
            },
            
            # 对比查询 -> 不过滤type，但需要多个机型
            PrimaryIntent.COMPARISON: {
                "type": ["对比", "参数", "规格"],
            }
        }
        
        # 二级意图到section的细化映射
        self.secondary_section_map = {
            # 参数查询
            SecondaryIntent.FLIGHT_PARAM: ["飞行参数", "飞行性能"],
            SecondaryIntent.CAMERA_PARAM: ["相机参数", "影像系统"],
            SecondaryIntent.BATTERY_PARAM: ["电池参数", "电源系统"],
            SecondaryIntent.TRANSMISSION_PARAM: ["图传参数", "传输系统"],
            SecondaryIntent.HARDWARE_SPEC: ["硬件规格", "物理规格"],
            
            # 操作步骤
            SecondaryIntent.TAKEOFF_LANDING: ["起飞降落", "起降操作"],
            SecondaryIntent.CALIBRATION: ["校准操作", "传感器校准"],
            SecondaryIntent.FIRMWARE_UPDATE: ["固件升级", "系统更新"],
            SecondaryIntent.ROUTE_PLANNING: ["航线规划", "任务规划"],
            
            # 故障排查
            SecondaryIntent.ERROR_CODE: ["错误代码", "故障代码"],
            SecondaryIntent.LOST_DRONE: ["飞丢找回", "失联处理"],
            
            # 维护保养
            SecondaryIntent.BATTERY_CARE: ["电池保养", "电池维护"],
            SecondaryIntent.PROPELLER_REPLACE: ["桨叶更换", "螺旋桨"],
        }
    
    def _init_model_keywords(self):
        """初始化机型关键词映射"""
        
        self.model_keyword_map = {
            # Mavic系列
            "mavic 4 pro": ["Mavic 4 Pro", "御4", "Mavic 4"],
            "mavic 3": ["Mavic 3", "御3"],
            "mavic 3 pro": ["Mavic 3 Pro", "御3 Pro"],
            "mavic air": ["Mavic Air", "Air"],
            "mavic mini": ["Mavic Mini", "Mini"],
            
            # Air系列
            "air 2": ["Air 2", "Air2"],
            "air 2s": ["Air 2S", "Air2S"],
            "air 3": ["Air 3", "Air3"],
            
            # Mini系列
            "mini 2": ["Mini 2", "Mini2"],
            "mini 3": ["Mini 3", "Mini3"],
            "mini 3 pro": ["Mini 3 Pro", "Mini3 Pro"],
            "mini 4 pro": ["Mini 4 Pro", "Mini4 Pro"],
            
            # Phantom系列
            "phantom": ["Phantom", "精灵"],
            "phantom 4": ["Phantom 4", "精灵4"],
            
            # Inspire系列
            "inspire": ["Inspire", "悟"],
            
            # FPV系列
            "fpv": ["FPV"],
            "avata": ["Avata"],
            
            # 农业机
            "t25": ["T25", "Agras T25"],
            "t25p": ["T25P", "Agras T25P"],
            "t40": ["T40", "Agras T40"],
            "t50": ["T50", "Agras T50"],
            "t60": ["T60", "Agras T60"],
            
            # 行业机
            "m30": ["M30", "Matrice 30"],
            "m300": ["M300", "Matrice 300"],
            "m350": ["M350", "Matrice 350"],
        }
    
    def _build_chroma_filter(
        self, 
        model_filters: Optional[List[str]] = None,
        type_filters: Optional[List[str]] = None,
        section_filters: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        构建ChromaDB过滤条件
        
        Args:
            model_filters: 机型过滤列表
            type_filters: 类型过滤列表
            section_filters: 章节过滤列表
            
        Returns:
            ChromaDB兼容的过滤字典
        """
        conditions = []
        
        # 机型过滤（多机型用OR）
        if model_filters:
            if len(model_filters) == 1:
                conditions.append({"model": {"$eq": model_filters[0]}})
            else:
                model_conditions = [{"model": {"$eq": m}} for m in model_filters]
                conditions.append({"$or": model_conditions})
        
        # 类型过滤
        if type_filters:
            if len(type_filters) == 1:
                conditions.append({"type": {"$eq": type_filters[0]}})
            else:
                type_conditions = [{"type": {"$eq": t}} for t in type_filters]
                conditions.append({"$or": type_conditions})
        
        # 章节过滤
        if section_filters:
            if len(section_filters) == 1:
                conditions.append({"section": {"$eq": section_filters[0]}})
            else:
                section_conditions = [{"section": {"$eq": s}} for s in section_filters]
                conditions.append({"$or": section_conditions})
        
        # 组合条件
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def generate(
        self, 
        query: str,
        intent: IntentClassification,
        detected_models: List[str]
    ) -> FilterCondition:
        """
        生成过滤条件
        
        Args:
            query: 查询文本
            intent: 意图分类结果
            detected_models: 检测到的机型列表
            
        Returns:
            FilterCondition对象
        """
        # 初始化过滤列表
        model_filters = []
        type_filters = []
        section_filters = []
        
        # 1. 基于意图生成类型和章节过滤
        if intent.primary_intent in self.intent_filters:
            intent_config = self.intent_filters[intent.primary_intent]
            
            if "type" in intent_config:
                type_filters.extend(intent_config["type"])
            
            if "section" in intent_config:
                section_filters.extend(intent_config["section"])
        
        # 2. 基于二级意图细化章节过滤
        if intent.secondary_intent in self.secondary_section_map:
            section_filters.extend(self.secondary_section_map[intent.secondary_intent])
        
        # 3. 基于检测到的机型生成机型过滤
        if detected_models:
            model_filters = detected_models
        
        # 4. 从查询中检测更多机型关键词
        query_lower = query.lower()
        for model_key, variants in self.model_keyword_map.items():
            for variant in variants:
                if variant.lower() in query_lower:
                    # 取第一个标准名作为过滤值
                    standard_name = variants[0]
                    if standard_name not in model_filters:
                        model_filters.append(standard_name)
                    break
        
        # 5. 特殊处理：对比查询需要保留所有相关机型
        if intent.primary_intent == PrimaryIntent.COMPARISON:
            # 对比查询不过滤类型，保留所有相关文档
            type_filters = []
            if len(model_filters) > 1:
                description = f"对比查询：机型 {', '.join(model_filters)}"
            else:
                description = "对比查询：未明确指定多个机型"
            priority = 2
        else:
            description = f"意图：{intent.primary_intent.value}"
            if intent.secondary_intent != SecondaryIntent.UNKNOWN:
                description += f" - {intent.secondary_intent.value}"
            priority = 1
        
        # 去重
        model_filters = list(set(model_filters))
        type_filters = list(set(type_filters))
        section_filters = list(set(section_filters))
        
        # 构建ChromaDB过滤条件
        chroma_filter = self._build_chroma_filter(
            model_filters=model_filters if model_filters else None,
            type_filters=type_filters if type_filters else None,
            section_filters=section_filters if section_filters else None
        )
        
        # 生成原始逻辑表达式（用于调试和日志）
        raw_parts = []
        if model_filters:
            if len(model_filters) == 1:
                raw_parts.append(f"model = '{model_filters[0]}'")
            else:
                model_str = ', '.join(["'" + m + "'" for m in model_filters])
                raw_parts.append(f"model IN ({model_str})")
        
        if type_filters:
            if len(type_filters) == 1:
                raw_parts.append(f"type = '{type_filters[0]}'")
            else:
                type_str = ', '.join(["'" + t + "'" for t in type_filters])
                raw_parts.append(f"type IN ({type_str})")
        
        if section_filters:
            if len(section_filters) == 1:
                raw_parts.append(f"section = '{section_filters[0]}'")
            else:
                section_str = ', '.join(["'" + s + "'" for s in section_filters])
                raw_parts.append(f"section IN ({section_str})")
        
        raw_expression = " AND ".join(raw_parts) if raw_parts else "无过滤条件"
        
        # 预期效果描述
        if chroma_filter:
            if model_filters:
                expected_effect = f"优先检索机型 {', '.join(model_filters[:2])} 的相关文档"
                if type_filters:
                    expected_effect += f"，类型为 {', '.join(type_filters[:2])}"
            else:
                expected_effect = f"检索类型为 {', '.join(type_filters[:2])} 的文档"
        else:
            expected_effect = "全局向量检索，无元数据过滤"
        
        return FilterCondition(
            raw_expression=raw_expression,
            chroma_filter=chroma_filter,
            priority=priority,
            description=description,
            expected_effect=expected_effect
        )
    
    def generate_from_keywords(
        self, 
        keywords: List[str],
        operator: FilterOperator = FilterOperator.OR
    ) -> FilterCondition:
        """
        基于关键词生成过滤条件
        
        Args:
            keywords: 关键词列表
            operator: 逻辑操作符
            
        Returns:
            FilterCondition对象
        """
        if not keywords:
            return FilterCondition(
                raw_expression="无过滤条件",
                chroma_filter=None,
                priority=3,
                description="无关键词过滤",
                expected_effect="全局向量检索"
            )
        
        # 构建关键词过滤（使用contains）
        keyword_conditions = []
        for kw in keywords:
            # 检查是否为机型
            is_model = False
            for model_key, variants in self.model_keyword_map.items():
                if kw in variants:
                    keyword_conditions.append({"model": {"$eq": variants[0]}})
                    is_model = True
                    break
            
            if not is_model:
                # 使用keywords字段包含
                keyword_conditions.append({"keywords": {"$contains": kw}})
        
        # 组合条件
        if len(keyword_conditions) == 1:
            chroma_filter = keyword_conditions[0]
        else:
            if operator == FilterOperator.AND:
                chroma_filter = {"$and": keyword_conditions}
            else:
                chroma_filter = {"$or": keyword_conditions}
        
        raw_expression = f"keywords {operator.value} ({', '.join(keywords)})"
        
        return FilterCondition(
            raw_expression=raw_expression,
            chroma_filter=chroma_filter,
            priority=2,
            description=f"关键词过滤：{', '.join(keywords[:3])}",
            expected_effect=f"检索包含关键词 {', '.join(keywords[:3])} 的文档"
        )


class FilterOptimizer:
    """过滤条件优化器"""
    
    def optimize(self, condition: FilterCondition, query_complexity: str = "normal") -> FilterCondition:
        """
        优化过滤条件
        
        Args:
            condition: 原始过滤条件
            query_complexity: 查询复杂度 (simple/normal/complex)
            
        Returns:
            优化后的过滤条件
        """
        # 简单查询：不过滤或轻量过滤
        if query_complexity == "simple":
            # 保留机型过滤，去除类型过滤以增加召回
            if condition.chroma_filter and "$and" in condition.chroma_filter:
                # 简化为仅机型过滤
                and_conditions = condition.chroma_filter["$and"]
                model_only = [c for c in and_conditions if "model" in c]
                if model_only:
                    return FilterCondition(
                        raw_expression=condition.raw_expression,
                        chroma_filter=model_only[0] if len(model_only) == 1 else {"$and": model_only},
                        priority=condition.priority,
                        description=condition.description + " (简化)",
                        expected_effect="轻量过滤，提高召回率"
                    )
        
        # 复杂查询：保持原有过滤
        return condition


# ============== 便捷函数 ==============

def generate_filter(
    query: str,
    intent_dict: Dict,
    detected_models: List[str]
) -> Dict:
    """
    便捷的过滤条件生成函数
    
    Args:
        query: 查询文本
        intent_dict: 意图分类结果字典
        detected_models: 检测到的机型
        
    Returns:
        过滤条件字典
    """
    from intent_classifier import IntentClassification, PrimaryIntent, SecondaryIntent
    
    # 解析意图
    primary = PrimaryIntent(intent_dict.get("primary_intent", "未知意图"))
    secondary = SecondaryIntent(intent_dict.get("secondary_intent", "未知"))
    
    intent = IntentClassification(
        primary_intent=primary,
        secondary_intent=secondary,
        confidence=intent_dict.get("confidence", 0.5)
    )
    
    generator = MetadataFilterGenerator()
    condition = generator.generate(query, intent, detected_models)
    
    return {
        "raw_expression": condition.raw_expression,
        "chroma_filter": condition.chroma_filter,
        "priority": condition.priority,
        "description": condition.description,
        "expected_effect": condition.expected_effect
    }


if __name__ == "__main__":
    # 测试
    from intent_classifier import HybridIntentClassifier
    
    test_cases = [
        {
            "query": "Mavic 4 Pro最大飞行高度是多少",
            "models": ["Mavic 4 Pro"]
        },
        {
            "query": "怎么进行指南针校准",
            "models": []
        },
        {
            "query": "Mavic 3和Mavic 4有什么区别",
            "models": ["Mavic 3", "Mavic 4 Pro"]
        },
    ]
    
    intent_classifier = HybridIntentClassifier(use_llm=False)
    filter_generator = MetadataFilterGenerator()
    
    print("=" * 80)
    print("元数据过滤条件生成测试")
    print("=" * 80)
    
    for case in test_cases:
        query = case["query"]
        models = case["models"]
        
        # 意图分类
        intent = intent_classifier.classify(query)
        
        # 生成过滤条件
        condition = filter_generator.generate(query, intent, models)
        
        print(f"\n查询: {query}")
        print(f"意图: {intent.primary_intent.value} - {intent.secondary_intent.value}")
        print(f"原始表达式: {condition.raw_expression}")
        print(f"ChromaDB过滤: {json.dumps(condition.chroma_filter, ensure_ascii=False, indent=2)}")
        print(f"优先级: {condition.priority}")
        print(f"描述: {condition.description}")
        print(f"预期效果: {condition.expected_effect}")
        print("-" * 80)
