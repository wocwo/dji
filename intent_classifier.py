"""
意图分类模块
功能：
1. 基于规则的快速意图分类（轻量级，无需LLM）
2. 基于LLM的精确意图分类（使用Prompt工程）

意图体系：
- 参数查询（飞行参数、相机参数、电池参数、图传参数、硬件规格）
- 操作步骤查询（起飞/降落、校准操作、功能开启、固件升级、航线规划）
- 故障排查查询（报错代码解析、故障现象排查、飞丢找回、异常处理）
- 功能说明查询（智能功能定义、使用限制、模式区别）
- 维护保养查询（电池保养、机身维护、桨叶更换、存储规范）
- 安全合规查询（禁飞规则、飞行须知、法律要求、风险提示）
- 对比查询（机型参数对比、功能模式对比）
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PrimaryIntent(Enum):
    """一级意图"""
    PARAMETER_QUERY = "参数查询"
    OPERATION_QUERY = "操作步骤查询"
    TROUBLESHOOTING = "故障排查查询"
    FUNCTION_QUERY = "功能说明查询"
    MAINTENANCE = "维护保养查询"
    SAFETY_COMPLIANCE = "安全合规查询"
    COMPARISON = "对比查询"
    UNKNOWN = "未知意图"


class SecondaryIntent(Enum):
    """二级意图"""
    # 参数查询
    FLIGHT_PARAM = "飞行参数"
    CAMERA_PARAM = "相机参数"
    BATTERY_PARAM = "电池参数"
    TRANSMISSION_PARAM = "图传参数"
    HARDWARE_SPEC = "硬件规格"
    
    # 操作步骤查询
    TAKEOFF_LANDING = "起飞降落"
    CALIBRATION = "校准操作"
    FEATURE_ENABLE = "功能开启"
    FIRMWARE_UPDATE = "固件升级"
    ROUTE_PLANNING = "航线规划"
    
    # 故障排查查询
    ERROR_CODE = "报错代码解析"
    FAULT_DIAGNOSIS = "故障现象排查"
    LOST_DRONE = "飞丢找回"
    ABNORMAL_HANDLING = "异常处理"
    
    # 功能说明查询
    SMART_FEATURE = "智能功能定义"
    USAGE_LIMITS = "使用限制"
    MODE_DIFFERENCE = "模式区别"
    
    # 维护保养查询
    BATTERY_CARE = "电池保养"
    BODY_MAINTENANCE = "机身维护"
    PROPELLER_REPLACE = "桨叶更换"
    STORAGE_GUIDE = "存储规范"
    
    # 安全合规查询
    NO_FLY_ZONE = "禁飞规则"
    FLIGHT_GUIDE = "飞行须知"
    LEGAL_REQ = "法律要求"
    RISK_WARNING = "风险提示"
    
    # 对比查询
    MODEL_COMPARISON = "机型参数对比"
    MODE_COMPARISON = "功能模式对比"
    
    UNKNOWN = "未知"


@dataclass
class IntentClassification:
    """意图分类结果"""
    primary_intent: PrimaryIntent
    secondary_intent: SecondaryIntent
    confidence: float
    
    # 额外信息
    detected_keywords: List[str] = field(default_factory=list)
    reasoning: str = ""  # 分类推理过程
    llm_used: bool = False  # 是否使用了LLM


class RuleBasedIntentClassifier:
    """基于规则的意图分类器"""
    
    def __init__(self):
        self._init_keyword_rules()
    
    def _init_keyword_rules(self):
        """初始化关键词规则"""
        
        self.intent_keywords = {
            # 参数查询
            PrimaryIntent.PARAMETER_QUERY: {
                SecondaryIntent.FLIGHT_PARAM: [
                    "飞行高度", "最大高度", "限高", "飞行速度", "最大速度", "续航",
                    "飞行时间", "航程", "飞行距离", "悬停时间", "抗风", "抗雨",
                    "工作温度", "海拔", "起飞海拔"
                ],
                SecondaryIntent.CAMERA_PARAM: [
                    "像素", "分辨率", "传感器", "CMOS", "光圈", "快门", "ISO",
                    "感光度", "焦距", "变焦", "广角", "长焦", "视频格式", "编码",
                    "帧率", "4K", "8K", "HDR", "D-Log", "色彩模式"
                ],
                SecondaryIntent.BATTERY_PARAM: [
                    "电池容量", "电量", "mAh", "电压", "循环次数", "充电时间",
                    "续航", "待机时间", "电池类型", "电芯", "智能电池"
                ],
                SecondaryIntent.TRANSMISSION_PARAM: [
                    "图传距离", "传输距离", "信号范围", "图传质量", "延迟",
                    "OcuSync", "O3+", "Wi-Fi", "5.8G", "2.4G", "频段"
                ],
                SecondaryIntent.HARDWARE_SPEC: [
                    "重量", "尺寸", "轴距", "桨叶", "电机", "电调", "IMU",
                    "指南针", "GPS", "视觉传感器", "避障", "结构", "材质"
                ]
            },
            
            # 操作步骤查询
            PrimaryIntent.OPERATION_QUERY: {
                SecondaryIntent.TAKEOFF_LANDING: [
                    "起飞", "降落", "返航", "紧急降落", "自动降落", "起飞准备",
                    "起飞检查", "降落点", "返航点"
                ],
                SecondaryIntent.CALIBRATION: [
                    "校准", "校磁", "指南针校准", "IMU校准", "云台校准",
                    "遥控器校准", "视觉校准", "水平校准"
                ],
                SecondaryIntent.FEATURE_ENABLE: [
                    "开启", "打开", "启用", "激活", "设置", "配置", "连接",
                    "配对", "绑定"
                ],
                SecondaryIntent.FIRMWARE_UPDATE: [
                    "升级", "更新", "固件", "刷机", "版本", "降级", "刷新"
                ],
                SecondaryIntent.BATTERY_CARE: [
                    "充电", "放电", "存储电量"
                ],
                SecondaryIntent.ROUTE_PLANNING: [
                    "航线", "航点", "规划", "任务", "航测", "测绘", "打点",
                    "区域", "航线设计"
                ]
            },
            
            # 故障排查查询
            PrimaryIntent.TROUBLESHOOTING: {
                SecondaryIntent.ERROR_CODE: [
                    "错误代码", "报错", "提示代码", "错误号", "故障码",
                    "Error", "Code", "错误信息"
                ],
                SecondaryIntent.FAULT_DIAGNOSIS: [
                    "不工作", "无法", "失败", "异常", "故障", "问题", "坏了",
                    "没反应", "失灵", "失效", "不能", "报错", "连不上", "连不上",
                    "连不到", "卡了", "卡顿", "延迟", "延迟高", "慢", "充不满",
                    "充不进", "充不了", "充不上", "不充电", "发热", "发烫",
                    "不稳定", "断断续续", "闪退", "死机"
                ],
                SecondaryIntent.LOST_DRONE: [
                    "飞丢", "丢失", "找不回", "失联", "断连", "坠毁", "坠落",
                    "掉落", "找回", "寻找", "找不到", "没了", "不见了"
                ],
                SecondaryIntent.ABNORMAL_HANDLING: [
                    "异常", "漂移", "抖动", "晃动", "震动", "偏移", "不稳",
                    "失控", "乱飞", "跑偏", "偏航", "掉高"
                ]
            },
            
            # 功能说明查询
            PrimaryIntent.FUNCTION_QUERY: {
                SecondaryIntent.SMART_FEATURE: [
                    "智能跟随", "ActiveTrack", "跟随模式", "兴趣点环绕", "POI",
                    "延时摄影", "全景", "大师镜头", "一键短片", "智能返航",
                    "避障", "感知", "APAS"
                ],
                SecondaryIntent.USAGE_LIMITS: [
                    "限制", "最大", "最小", "范围", "条件", "要求", "注意",
                    "不适合", "禁止"
                ],
                SecondaryIntent.MODE_DIFFERENCE: [
                    "区别", "差异", "不同", "模式", "档位", "Cine", "Normal",
                    "Sport", "运动模式", "平稳模式"
                ]
            },
            
            # 维护保养查询
            PrimaryIntent.MAINTENANCE: {
                SecondaryIntent.BATTERY_CARE: [
                    "电池保养", "存储电量", "放电", "电池维护", "电池寿命",
                    "长期存放", "电池健康"
                ],
                SecondaryIntent.BODY_MAINTENANCE: [
                    "清洁", "保养", "维护", "检查", "擦拭", "防水", "防尘",
                    "存储环境", "存放"
                ],
                SecondaryIntent.PROPELLER_REPLACE: [
                    "换桨", "桨叶更换", "螺旋桨", "安装桨叶", "拆卸桨叶",
                    "桨叶型号", "备用桨"
                ],
                SecondaryIntent.STORAGE_GUIDE: [
                    "存储", "存放", "保管", "收纳", "包装", "运输", "携带",
                    "长期不用", "闲置"
                ]
            },
            
            # 安全合规查询
            PrimaryIntent.SAFETY_COMPLIANCE: {
                SecondaryIntent.NO_FLY_ZONE: [
                    "禁飞", "限飞区", "禁飞区", "GEO", "地理围栏", "机场",
                    "军事区", "限制区域"
                ],
                SecondaryIntent.FLIGHT_GUIDE: [
                    "安全", "注意", "须知", "提醒", "警告", " caution",
                    "warning", "安全第一"
                ],
                SecondaryIntent.LEGAL_REQ: [
                    "法规", "法律", "规定", "实名", "登记", "注册", "执照",
                    "飞行许可", "UOM", "民航局"
                ],
                SecondaryIntent.RISK_WARNING: [
                    "风险", "危险", "隐患", "事故", "保险", "责任", "赔偿"
                ]
            },
            
            # 对比查询
            PrimaryIntent.COMPARISON: {
                SecondaryIntent.MODEL_COMPARISON: [
                    "对比", "比较", "区别", "vs", "versus", "哪个好",
                    "推荐", "选择", "选型"
                ],
                SecondaryIntent.MODE_COMPARISON: [
                    "模式对比", "对比模式", "比较模式", "模式区别"
                ]
            }
        }
        
        # 优先级排序（某些关键词更确定属于某个意图）
        self.priority_keywords = {
            # 故障排查 - 高优先级
            "飞丢": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.LOST_DRONE, 1.0),
            "丢失": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.LOST_DRONE, 0.9),
            "连不上": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "连接失败": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "连不到": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "卡了": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "卡顿": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "充不满": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "充不进": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "不充电": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "发热": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.9),
            "坏了": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            "报错": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.ERROR_CODE, 0.9),
            "问题": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.85),
            "故障": (PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.FAULT_DIAGNOSIS, 0.95),
            
            # 安全合规
            "禁飞": (PrimaryIntent.SAFETY_COMPLIANCE, SecondaryIntent.NO_FLY_ZONE, 1.0),
            
            # 对比查询
            "对比": (PrimaryIntent.COMPARISON, SecondaryIntent.MODEL_COMPARISON, 1.0),
            "区别": (PrimaryIntent.COMPARISON, SecondaryIntent.MODEL_COMPARISON, 0.9),
            "vs": (PrimaryIntent.COMPARISON, SecondaryIntent.MODEL_COMPARISON, 0.95),
            
            # 操作步骤
            "校准": (PrimaryIntent.OPERATION_QUERY, SecondaryIntent.CALIBRATION, 0.95),
            "升级": (PrimaryIntent.OPERATION_QUERY, SecondaryIntent.FIRMWARE_UPDATE, 0.9),
            "航线": (PrimaryIntent.OPERATION_QUERY, SecondaryIntent.ROUTE_PLANNING, 0.95),
            
            # 维护保养 - 优先级高于一般操作词
            "保养": (PrimaryIntent.MAINTENANCE, SecondaryIntent.BODY_MAINTENANCE, 0.95),
            "维护": (PrimaryIntent.MAINTENANCE, SecondaryIntent.BODY_MAINTENANCE, 0.9),
            
            # 辅助判断词（优先级较低）
            "怎么": (PrimaryIntent.OPERATION_QUERY, SecondaryIntent.UNKNOWN, 0.5),
            "如何": (PrimaryIntent.OPERATION_QUERY, SecondaryIntent.UNKNOWN, 0.5),
        }
    
    def classify(self, query: str) -> IntentClassification:
        """
        基于规则进行意图分类
        
        Args:
            query: 清洗后的查询文本
            
        Returns:
            IntentClassification对象
        """
        query_lower = query.lower()
        detected_keywords = []
        
        # 1. 检查高优先级关键词
        for keyword, (pri, sec, conf) in self.priority_keywords.items():
            if keyword in query:
                return IntentClassification(
                    primary_intent=pri,
                    secondary_intent=sec,
                    confidence=conf,
                    detected_keywords=[keyword],
                    reasoning=f"匹配高优先级关键词: {keyword}",
                    llm_used=False
                )
        
        # 2. 统计各意图类别的关键词命中数
        intent_scores = {}
        
        for primary, secondary_dict in self.intent_keywords.items():
            for secondary, keywords in secondary_dict.items():
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in query:
                        score += 1
                        matched_keywords.append(keyword)
                        detected_keywords.append(keyword)
                
                if score > 0:
                    intent_scores[(primary, secondary)] = {
                        "score": score,
                        "keywords": matched_keywords
                    }
        
        # 3. 选择得分最高的意图
        if intent_scores:
            best_match = max(intent_scores.items(), key=lambda x: x[1]["score"])
            (primary, secondary), info = best_match
            
            # 计算置信度（基于关键词匹配数量）
            confidence = min(0.9, 0.5 + info["score"] * 0.1)
            
            return IntentClassification(
                primary_intent=primary,
                secondary_intent=secondary,
                confidence=confidence,
                detected_keywords=list(set(info["keywords"])),
                reasoning=f"关键词匹配: {info['keywords']}",
                llm_used=False
            )
        
        # 4. 处理特殊模式查询
        # 4.1 "怎么/如何 + 动词" -> 操作步骤查询
        operation_patterns = ["怎么", "如何", "咋", "怎样"]
        if any(p in query for p in operation_patterns):
            return IntentClassification(
                primary_intent=PrimaryIntent.OPERATION_QUERY,
                secondary_intent=SecondaryIntent.UNKNOWN,
                confidence=0.6,
                detected_keywords=[],
                reasoning="匹配操作步骤查询模式(怎么/如何/咋)",
                llm_used=False
            )
        
        # 4.2 "为什么 + 问题" -> 故障排查查询
        if "为什么" in query or "为啥" in query:
            return IntentClassification(
                primary_intent=PrimaryIntent.TROUBLESHOOTING,
                secondary_intent=SecondaryIntent.FAULT_DIAGNOSIS,
                confidence=0.7,
                detected_keywords=[],
                reasoning="匹配故障排查查询模式(为什么/为啥)",
                llm_used=False
            )
        
        # 4.3 "怎么办" -> 故障排查查询
        if "怎么办" in query:
            # 检查前面是否有故障描述
            return IntentClassification(
                primary_intent=PrimaryIntent.TROUBLESHOOTING,
                secondary_intent=SecondaryIntent.FAULT_DIAGNOSIS,
                confidence=0.7,
                detected_keywords=[],
                reasoning="匹配故障排查查询模式(怎么办)",
                llm_used=False
            )
        
        # 5. 无法分类
        return IntentClassification(
            primary_intent=PrimaryIntent.UNKNOWN,
            secondary_intent=SecondaryIntent.UNKNOWN,
            confidence=0.3,
            detected_keywords=[],
            reasoning="未匹配到明确关键词",
            llm_used=False
        )


class LLMIntentClassifier:
    """基于LLM的意图分类器"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        """
        初始化LLM分类器
        
        Args:
            api_key: API密钥
            api_base: API基础URL
        """
        self.api_key = api_key
        self.api_base = api_base or "http://localhost:11434"
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """检查LLM是否可用"""
        try:
            # 尝试导入ollama或openai
            try:
                import openai
                self.client = openai.OpenAI(
                    api_key=self.api_key or "ollama",
                    base_url=f"{self.api_base}/v1"
                )
                self.model = "deepseek-r1:7b"  # 默认模型
                return True
            except ImportError:
                return False
        except Exception:
            return False
    
    def _build_prompt(self, query: str) -> str:
        """构建分类Prompt"""
        
        primary_intents = [
            "参数查询（飞行参数、相机参数、电池参数、图传参数、硬件规格）",
            "操作步骤查询（起飞/降落、校准操作、功能开启、固件升级、航线规划）",
            "故障排查查询（报错代码解析、故障现象排查、飞丢找回、异常处理）",
            "功能说明查询（智能功能定义、使用限制、模式区别）",
            "维护保养查询（电池保养、机身维护、桨叶更换、存储规范）",
            "安全合规查询（禁飞规则、飞行须知、法律要求、风险提示）",
            "对比查询（机型参数对比、功能模式对比）"
        ]
        
        prompt = f"""你是一位专业的无人机领域意图分类专家。请分析用户的查询，将其分类到预定义的意图类别中。

用户查询：{query}

可选的一级意图类别：
{chr(10).join([f"{i+1}. {intent}" for i, intent in enumerate(primary_intents)])}

请输出JSON格式结果：
{{
    "primary_intent": "一级意图名称（从上述列表中选择）",
    "secondary_intent": "二级意图名称",
    "confidence": 0.95,
    "reasoning": "分类推理过程"
}}

注意：
1. primary_intent 必须从上述列表中选择
2. secondary_intent 应该是一级意图下的细分
3. confidence 范围0-1，表示置信度
4. reasoning 简要说明分类理由

请只输出JSON，不要其他内容。"""
        
        return prompt
    
    def classify(self, query: str) -> Optional[IntentClassification]:
        """
        使用LLM进行意图分类
        
        Args:
            query: 清洗后的查询文本
            
        Returns:
            IntentClassification对象，如果失败返回None
        """
        if not self.available:
            return None
        
        try:
            prompt = self._build_prompt(query)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的无人机领域意图分类助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # 解析JSON
            try:
                # 尝试提取JSON部分
                json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                
                result = json.loads(content)
                
                # 映射到枚举
                primary_str = result.get("primary_intent", "")
                secondary_str = result.get("secondary_intent", "")
                confidence = result.get("confidence", 0.5)
                reasoning = result.get("reasoning", "")
                
                # 尝试匹配枚举
                primary = PrimaryIntent.UNKNOWN
                for p in PrimaryIntent:
                    if p.value in primary_str or primary_str in p.value:
                        primary = p
                        break
                
                secondary = SecondaryIntent.UNKNOWN
                for s in SecondaryIntent:
                    if s.value in secondary_str or secondary_str in s.value:
                        secondary = s
                        break
                
                return IntentClassification(
                    primary_intent=primary,
                    secondary_intent=secondary,
                    confidence=confidence,
                    detected_keywords=[],
                    reasoning=reasoning,
                    llm_used=True
                )
                
            except json.JSONDecodeError:
                return None
                
        except Exception as e:
            print(f"LLM分类失败: {e}")
            return None


class HybridIntentClassifier:
    """混合意图分类器（规则+LLM）"""
    
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None):
        """
        初始化混合分类器
        
        Args:
            use_llm: 是否使用LLM增强
            api_key: LLM API密钥
        """
        self.rule_classifier = RuleBasedIntentClassifier()
        
        self.llm_classifier = None
        if use_llm:
            self.llm_classifier = LLMIntentClassifier(api_key=api_key)
    
    def classify(self, query: str, prefer_llm: bool = False) -> IntentClassification:
        """
        进行意图分类
        
        Args:
            query: 查询文本
            prefer_llm: 优先使用LLM（如果可用）
            
        Returns:
            IntentClassification对象
        """
        # 先使用规则分类
        rule_result = self.rule_classifier.classify(query)
        
        # 如果规则分类置信度高，直接返回
        if rule_result.confidence >= 0.9:
            return rule_result
        
        # 尝试LLM分类（如果可用且需要）
        if self.llm_classifier and (prefer_llm or rule_result.confidence < 0.7):
            llm_result = self.llm_classifier.classify(query)
            
            if llm_result and llm_result.confidence > rule_result.confidence:
                return llm_result
        
        return rule_result
    
    def batch_classify(self, queries: List[str]) -> List[IntentClassification]:
        """批量分类"""
        return [self.classify(q) for q in queries]


# ============== 便捷函数 ==============

def classify_intent(query: str, use_llm: bool = False) -> Dict:
    """
    便捷的意图分类函数
    
    Args:
        query: 用户查询
        use_llm: 是否使用LLM
        
    Returns:
        分类结果字典
    """
    classifier = HybridIntentClassifier(use_llm=use_llm)
    result = classifier.classify(query)
    
    return {
        "primary_intent": result.primary_intent.value,
        "secondary_intent": result.secondary_intent.value,
        "confidence": result.confidence,
        "detected_keywords": result.detected_keywords,
        "reasoning": result.reasoning,
        "llm_used": result.llm_used
    }


if __name__ == "__main__":
    # 测试
    test_queries = [
        "Mavic 4 Pro最大飞行高度是多少",
        "怎么进行指南针校准",
        "飞机飞丢了怎么找回",
        "智能跟随功能怎么用",
        "电池如何保养",
        "禁飞区有哪些",
        "Mavic 3和Mavic 4有什么区别",
    ]
    
    classifier = HybridIntentClassifier(use_llm=False)
    
    print("=" * 80)
    print("意图分类测试")
    print("=" * 80)
    
    for query in test_queries:
        result = classifier.classify(query)
        print(f"\n查询: {query}")
        print(f"一级意图: {result.primary_intent.value}")
        print(f"二级意图: {result.secondary_intent.value}")
        print(f"置信度: {result.confidence}")
        print(f"关键词: {result.detected_keywords}")
        print(f"推理: {result.reasoning}")
        print("-" * 80)
