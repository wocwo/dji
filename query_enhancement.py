"""
查询理解与增强模块
功能：
1. 查询清洗 - 去除无效信息、统一格式
2. 非标表述修正 - 口语化转标准术语
3. 查询扩展 - 同义词扩展用于召回
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class QueryType(Enum):
    """查询类型"""
    STANDARD = "standard"      # 标准查询
    ABBREVIATION = "abbreviation"  # 含缩写
    TYPO = "typo"             # 含错别字
    MULTI_INTENT = "multi_intent"  # 多意图
    VAGUE = "vague"           # 模糊查询
    INVALID = "invalid"       # 无效查询


@dataclass
class EnhancedQuery:
    """增强后的查询结果"""
    original_query: str                    # 原始查询
    cleaned_query: str                     # 清洗后查询
    query_type: QueryType                  # 查询类型
    
    # 2类增强查询
    precise_query: str                     # 精准检索查询（标准术语）
    expanded_queries: List[str]            # 召回扩展查询列表（含同义词）
    
    # 元数据
    detected_abbreviations: Dict[str, str] = field(default_factory=dict)  # 检测到的缩写
    detected_typos: Dict[str, str] = field(default_factory=dict)          # 检测到的错别字
    detected_models: List[str] = field(default_factory=list)              # 检测到的机型
    confidence: float = 1.0                                               # 处理置信度


class QueryEnhancer:
    """查询增强器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化查询增强器
        
        Args:
            config_path: 配置文件路径，包含术语库、同义词库等
        """
        self._load_terminology_databases()
        self._compile_patterns()
    
    def _load_terminology_databases(self):
        """加载术语数据库"""
        
        # 大疆产品术语库 - 口语化表述 -> 标准术语
        self.terminology_map = {
            # 基础操作
            "怎么飞": "起飞操作",
            "咋飞": "起飞操作",
            "如何起飞": "起飞操作",
            "咋起飞": "起飞操作",
            "怎么降落": "降落操作",
            "咋降落": "降落操作",
            "怎么下来": "降落操作",
            "咋下来": "降落操作",
            "怎么充电": "电池充电",
            "咋充电": "电池充电",
            "怎么关": "关机操作",
            "咋关": "关机操作",
            "怎么开": "开机操作",
            "咋开": "开机操作",
            
            # 遥控器相关
            "遥控手柄": "遥控器",
            "遥控器配对": "遥控器配对",
            "遥控器连接失败": "遥控器连接失败",
            "遥控器无响应": "遥控器无响应",
            "怎么连接遥控": "遥控器配对",
            "咋连遥控": "遥控器配对",
            "遥控连不上": "遥控器连接失败",
            "遥控没反应": "遥控器无响应",
            "遥控": "遥控器",
            "手柄": "遥控器",
            "控": "遥控器",
            
            # 图传相关
            "图传系统": "图传系统",
            "图传画面": "图传画面",
            "图传卡顿": "图传卡顿",
            "图传延迟": "图传延迟",
            "图传": "图传",
            "画面卡顿": "画面卡顿",
            "画面延迟": "画面延迟",
            "画面": "图传画面",
            "视频传输": "视频传输",
            "视频": "视频传输",
            "信号不好": "信号弱",
            "没信号": "信号丢失",
            
            # 飞行相关
            "飞丢": "飞行器丢失",
            "丢了": "飞行器丢失",
            "找不到了": "飞行器丢失",
            "飞丢了": "飞行器丢失",
            "飞不起来": "无法起飞",
            "飞不上去": "无法起飞",
            "不能飞": "无法起飞",
            
            # 校准相关
            "校准操作": "校准操作",
            "指南针校准": "指南针校准",
            "IMU校准": "IMU校准",
            "指南针异常": "指南针异常",
            "校准": "校准操作",
            "校磁": "指南针校准",
            "磁罗盘": "指南针",
            "指南针不准": "指南针异常",
            "陀螺仪": "IMU校准",
            "校IMU": "IMU校准",
            
            # 避障相关
            "避障": "避障功能",
            "障碍物": "避障功能",
            "防撞": "避障功能",
            "会撞吗": "避障检测",
            "避障不管用": "避障失效",
            
            # 电池相关
            "智能飞行电池": "智能飞行电池",
            "电池耗尽": "电池耗尽",
            "充电异常": "充电异常",
            "充电缓慢": "充电缓慢",
            "电池膨胀": "电池膨胀",
            "电池": "电池",
            "电": "电量",
            "没电": "电池耗尽",
            "充不满": "充电异常",
            "充电慢": "充电缓慢",
            "电池鼓包": "电池膨胀",
            
            # 相机相关
            "拍照": "拍摄照片",
            "录像": "录制视频",
            "照相": "拍摄照片",
            "拍视频": "录制视频",
            "摄像头": "相机",
            "云台": "云台相机",
            "抖": "云台抖动",
            "模糊": "图像模糊",
            
            # 固件相关
            "刷机": "固件刷新",
            "固件版本": "固件版本",
            "固件升级": "固件升级",
            "固件更新": "固件更新",
            "升级": "固件升级",
            "更新": "固件更新",
            "固件": "固件",
            "版本": "固件版本",
            
            # 故障相关
            "坏了": "设备故障",
            "出问题了": "设备异常",
            "报错": "错误代码",
            "提示": "系统提示",
            "红灯": "状态指示灯红色",
            "黄灯": "状态指示灯黄色",
            "绿灯": "状态指示灯绿色",
            
            # 航线相关
            "航线规划": "航线规划",
            "航点设置": "航点设置",
            "标记航点": "标记航点",
            "自动飞行": "自动飞行",
            "航线": "航线",
            "航点": "航点",
            "打点": "标记航点",
            "规划路线": "航线规划",
            "自动飞": "自动飞行",
        }
        
        # 同义词扩展库 - 用于召回扩展
        self.synonym_map = {
            # 基础术语
            "起飞": ["起飞", "升空", "离地", "开始飞行"],
            "降落": ["降落", "着陆", "下降", "返航降落"],
            "返航": ["返航", "自动返航", "RTH", "Return to Home", "返回起飞点"],
            "悬停": ["悬停", "定点悬停", "空中悬停", "保持位置"],
            
            # 硬件
            "遥控器": ["遥控器", "发射器", "控制手柄", "RC", "Radio Controller"],
            "电池": ["电池", "智能飞行电池", "动力电池", "电源"],
            "电机": ["电机", "马达", "电调", "动力系统"],
            "桨叶": ["桨叶", "螺旋桨", "机翼"],
            
            # 传感器
            "指南针": ["指南针", "磁罗盘", "磁力计", "Compass"],
            "IMU": ["IMU", "惯性测量单元", "惯性传感器", "姿态传感器"],
            "GPS": ["GPS", "全球定位系统", "卫星定位", "GNSS"],
            "视觉": ["视觉", "视觉系统", "视觉传感器", "摄像头"],
            
            # 功能
            "避障": ["避障", "障碍物感知", "障碍检测", "视觉避障"],
            "跟随": ["跟随", "智能跟随", "Follow Me", "目标跟踪"],
            "环绕": ["环绕", "兴趣点环绕", "POI", "Point of Interest"],
            "延时": ["延时", "延时摄影", "Time Lapse"],
            
            # 图传
            "图传": ["图传", "图传系统", "图像传输", "视频传输", "FPV"],
            "信号": ["信号", "连接质量", "链路质量", "传输质量"],
            "卡顿": ["卡顿", "延迟", "丢帧", "画面不流畅"],
            
            # 相机参数
            "分辨率": ["分辨率", "像素", "图像尺寸", "画面大小"],
            "帧率": ["帧率", "FPS", "拍摄帧率", "视频帧率"],
            "ISO": ["ISO", "感光度", "感光值"],
            "快门": ["快门", "快门速度", "曝光时间"],
            "光圈": ["光圈", "光圈值", "F值"],
        }
        
        # 机型识别库
        self.drone_models = [
            "Mavic 4 Pro", "Mavic 3", "Mavic 3 Pro", "Mavic 3 Classic",
            "Mavic Air 2", "Mavic Air 2S", "Mavic Mini", "Mini 2", "Mini 3", "Mini 3 Pro", "Mini 4 Pro",
            "Air 2", "Air 2S", "Air 3",
            "Phantom 4", "Phantom 4 Pro", "Phantom 4 Advanced", "Phantom 3",
            "Inspire 2", "Inspire 3",
            "FPV", "Avata", "Avata 2",
            "T25", "T25P", "T40", "T50", "T60",
            "Agras", "Matrice", "M30", "M30T", "M350", "M300",
            "Mini 2 SE", "Mini 4K",
        ]
        
        # 常见错别字映射
        self.typo_map = {
            "摇控": "遥控",
            "图象": "图像",
            "摄象": "摄像",
            "像机": "相机",
            "电沲": "电池",
            "充点": "充电",
            "飞鸡": "飞机",
            "无入机": "无人机",
            "遥控哭": "遥控器",
            "桨叶": "桨叶",
            "罗盤": "罗盘",
            "指北针": "指南针",
            "云抬": "云台",
            "固件": "固件",
            "升及": "升级",
            "更心": "更新",
        }
        
        # 无效词汇列表
        self.invalid_patterns = [
            r'[你好啊呀呢吧]|请问|麻烦|帮忙|谢谢|感谢|请',  # 语气词、敬语
            r'[😀-🿿]|[^\u4e00-\u9fa5a-zA-Z0-9\s]',  # emoji和特殊符号
        ]
    
    def _compile_patterns(self):
        """编译正则表达式"""
        self.invalid_regex = re.compile('|'.join(self.invalid_patterns))
        self.space_regex = re.compile(r'\s+')
        
    def clean_query(self, query: str) -> Tuple[str, QueryType]:
        """
        清洗查询文本
        
        Args:
            query: 原始查询
            
        Returns:
            (清洗后查询, 查询类型)
        """
        if not query or not query.strip():
            return "", QueryType.INVALID
        
        # 1. 去除前后空格
        cleaned = query.strip()
        
        # 2. 统一换行和空格
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        cleaned = self.space_regex.sub(' ', cleaned)
        
        # 3. 去除无效信息（语气词、敬语、特殊符号）
        cleaned = self.invalid_regex.sub('', cleaned)
        
        # 4. 再次清理空格
        cleaned = self.space_regex.sub(' ', cleaned).strip()
        
        # 判断查询类型
        query_type = self._detect_query_type(query, cleaned)
        
        return cleaned, query_type
    
    def _detect_query_type(self, original: str, cleaned: str) -> QueryType:
        """检测查询类型"""
        if not cleaned:
            return QueryType.INVALID
        
        # 检查是否包含错别字
        for typo in self.typo_map.keys():
            if typo in cleaned:
                return QueryType.TYPO
        
        # 检查是否包含多意图（通过连接词判断）
        multi_intent_markers = ['同时', '还有', '以及', '并且', '另外', '顺便', '和', '与']
        if any(marker in cleaned for marker in multi_intent_markers):
            # 同时检查是否有实际的操作意图
            return QueryType.MULTI_INTENT
        
        # 检查是否包含缩写
        for abbrev in self.terminology_map.keys():
            if abbrev in cleaned and len(abbrev) <= 4:  # 短词可能是缩写
                return QueryType.ABBREVIATION
        
        # 检查是否模糊（过短或过于宽泛）
        if len(cleaned) <= 3 or cleaned in ['怎么', '什么', '如何', '为什么']:
            return QueryType.VAGUE
        
        return QueryType.STANDARD
    
    def correct_typos(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        纠正错别字
        
        Args:
            query: 输入查询
            
        Returns:
            (纠正后查询, 纠正映射字典)
        """
        corrected = query
        corrections = {}
        
        for typo, correct in self.typo_map.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correct)
                corrections[typo] = correct
        
        return corrected, corrections
    
    def standardize_terminology(self, query: str) -> Tuple[str, Dict[str, str]]:
        """
        标准化术语（口语化转标准术语）
        
        Args:
            query: 输入查询
            
        Returns:
            (标准化后查询, 术语映射字典)
        """
        # 使用占位符避免重复替换问题
        placeholder_map = {}
        placeholder_counter = 0
        mappings = {}
        
        # 按长度降序排序，优先匹配长词
        sorted_terms = sorted(self.terminology_map.items(), key=lambda x: len(x[0]), reverse=True)
        
        # 第一步：用占位符替换所有匹配项
        result = query
        for colloquial, standard in sorted_terms:
            if colloquial in result:
                placeholder = f"__PLACEHOLDER_{placeholder_counter}__"
                placeholder_counter += 1
                result = result.replace(colloquial, placeholder)
                placeholder_map[placeholder] = standard
                mappings[colloquial] = standard
        
        # 第二步：将占位符替换为标准术语
        for placeholder, standard in placeholder_map.items():
            result = result.replace(placeholder, standard)
        
        return result, mappings
    
    def extract_models(self, query: str) -> List[str]:
        """
        提取查询中的机型信息
        
        Args:
            query: 输入查询
            
        Returns:
            检测到的机型列表
        """
        detected = []
        for model in self.drone_models:
            if model.lower() in query.lower():
                detected.append(model)
        
        # 特殊处理一些简称
        query_lower = query.lower()
        if 'mavic' in query_lower or '御' in query:
            if '御3' in query and "Mavic 3" not in detected:
                detected.append("Mavic 3")
            if '御4' in query and "Mavic 4" not in detected:
                detected.append("Mavic 4 Pro")
        
        return list(set(detected))  # 去重
    
    def expand_query(self, query: str) -> List[str]:
        """
        扩展查询（同义词扩展）
        
        Args:
            query: 输入查询
            
        Returns:
            扩展后的查询列表
        """
        expanded_queries = [query]  # 原始查询
        
        # 查找需要扩展的术语
        for term, synonyms in self.synonym_map.items():
            if term in query:
                # 为每个同义词生成新查询
                for synonym in synonyms:
                    if synonym != term:
                        expanded = query.replace(term, synonym)
                        if expanded not in expanded_queries:
                            expanded_queries.append(expanded)
        
        return expanded_queries
    
    def enhance(self, query: str) -> EnhancedQuery:
        """
        执行完整的查询增强流程
        
        Args:
            query: 原始用户查询
            
        Returns:
            EnhancedQuery对象
        """
        # 1. 清洗查询
        cleaned, query_type = self.clean_query(query)
        
        # 如果无效，直接返回
        if query_type == QueryType.INVALID:
            return EnhancedQuery(
                original_query=query,
                cleaned_query=cleaned,
                query_type=query_type,
                precise_query=cleaned,
                expanded_queries=[cleaned] if cleaned else [],
                confidence=0.0
            )
        
        # 2. 纠正错别字
        typo_corrected, typo_corrections = self.correct_typos(cleaned)
        
        # 3. 标准化术语（生成精准查询）
        precise_query, term_mappings = self.standardize_terminology(typo_corrected)
        
        # 4. 提取机型信息
        detected_models = self.extract_models(query)
        
        # 5. 生成扩展查询（用于召回扩展）
        expanded_queries = self.expand_query(precise_query)
        
        # 6. 计算置信度
        confidence = self._calculate_confidence(
            query_type, typo_corrections, term_mappings
        )
        
        return EnhancedQuery(
            original_query=query,
            cleaned_query=cleaned,
            query_type=query_type,
            precise_query=precise_query,
            expanded_queries=expanded_queries,
            detected_abbreviations=term_mappings,
            detected_typos=typo_corrections,
            detected_models=detected_models,
            confidence=confidence
        )
    
    def _calculate_confidence(
        self, 
        query_type: QueryType, 
        typo_corrections: Dict[str, str],
        term_mappings: Dict[str, str]
    ) -> float:
        """计算处理置信度"""
        base_confidence = 1.0
        
        # 根据查询类型调整
        if query_type == QueryType.INVALID:
            return 0.0
        elif query_type == QueryType.TYPO:
            base_confidence -= 0.1
        elif query_type == QueryType.VAGUE:
            base_confidence -= 0.2
        elif query_type == QueryType.MULTI_INTENT:
            base_confidence -= 0.1
        
        # 根据修正数量调整
        if typo_corrections:
            base_confidence -= len(typo_corrections) * 0.05
        
        return max(0.5, min(1.0, base_confidence))
    
    def batch_enhance(self, queries: List[str]) -> List[EnhancedQuery]:
        """
        批量增强查询
        
        Args:
            queries: 查询列表
            
        Returns:
            增强结果列表
        """
        return [self.enhance(q) for q in queries]


# ============== 便捷函数 ==============

def enhance_query(query: str) -> Dict:
    """
    便捷的查询增强函数
    
    Args:
        query: 用户查询
        
    Returns:
        增强结果字典
    """
    enhancer = QueryEnhancer()
    result = enhancer.enhance(query)
    
    return {
        "original": result.original_query,
        "cleaned": result.cleaned_query,
        "precise": result.precise_query,
        "expanded": result.expanded_queries,
        "query_type": result.query_type.value,
        "models": result.detected_models,
        "abbreviations": result.detected_abbreviations,
        "typos": result.detected_typos,
        "confidence": result.confidence
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
    ]
    
    enhancer = QueryEnhancer()
    
    print("=" * 80)
    print("查询增强测试")
    print("=" * 80)
    
    for query in test_queries:
        result = enhancer.enhance(query)
        print(f"\n原始查询: {result.original_query}")
        print(f"清洗结果: {result.cleaned_query}")
        print(f"精准查询: {result.precise_query}")
        print(f"扩展查询: {result.expanded_queries}")
        print(f"查询类型: {result.query_type.value}")
        print(f"检测到机型: {result.detected_models}")
        print(f"置信度: {result.confidence}")
        print("-" * 80)
