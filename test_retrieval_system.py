"""
检索系统测试与评估模块

测试覆盖：
1. 标准查询
2. 含缩写的查询
3. 含错别字的查询
4. 多意图查询
5. 模糊查询
6. 无效查询

评估指标：
- 意图识别准确率
- 查询改写准确率
- 过滤条件生成准确率
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from retrieval_system import IntelligentRetrievalSystem, RetrievalEvaluator
from intent_classifier import PrimaryIntent, SecondaryIntent


@dataclass
class TestCase:
    """测试用例"""
    name: str
    query: str
    category: str  # standard/abbreviation/typo/multi_intent/vague/invalid
    
    # 期望结果
    expected_primary_intent: PrimaryIntent
    expected_keywords_in_rewrite: List[str]  # 改写结果应包含的关键词
    expected_has_filter: bool
    
    # 可选：更详细的期望
    expected_secondary_intent: SecondaryIntent = None
    expected_models: List[str] = None


class RetrievalSystemTester:
    """检索系统测试器"""
    
    def __init__(self):
        self.system = IntelligentRetrievalSystem(use_llm_for_intent=False)
        self.evaluator = RetrievalEvaluator()
        
        # 测试结果
        self.test_results = []
    
    def get_test_cases(self) -> List[TestCase]:
        """获取测试用例集"""
        
        test_cases = [
            # ========== 标准查询 ==========
            TestCase(
                name="标准参数查询",
                query="Mavic 4 Pro最大飞行高度是多少",
                category="standard",
                expected_primary_intent=PrimaryIntent.PARAMETER_QUERY,
                expected_keywords_in_rewrite=["Mavic 4 Pro", "飞行高度"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.FLIGHT_PARAM,
                expected_models=["Mavic 4 Pro"]
            ),
            TestCase(
                name="标准操作查询",
                query="如何进行指南针校准",
                category="standard",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["指南针", "校准"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.CALIBRATION
            ),
            TestCase(
                name="标准功能查询",
                query="智能跟随功能怎么用",
                category="standard",
                expected_primary_intent=PrimaryIntent.FUNCTION_QUERY,
                expected_keywords_in_rewrite=["智能跟随"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.SMART_FEATURE
            ),
            
            # ========== 含缩写的查询 ==========
            TestCase(
                name="缩写_起飞",
                query="咋起飞",
                category="abbreviation",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["起飞"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.TAKEOFF_LANDING
            ),
            TestCase(
                name="缩写_充电",
                query="电池咋充电",
                category="abbreviation",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["电池", "充电"],
                expected_has_filter=True
            ),
            TestCase(
                name="缩写_遥控器",
                query="遥控连不上",
                category="abbreviation",
                expected_primary_intent=PrimaryIntent.TROUBLESHOOTING,
                expected_keywords_in_rewrite=["遥控器", "连接"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.FAULT_DIAGNOSIS
            ),
            TestCase(
                name="缩写_图传",
                query="图传卡了",
                category="abbreviation",
                expected_primary_intent=PrimaryIntent.TROUBLESHOOTING,
                expected_keywords_in_rewrite=["图传"],
                expected_has_filter=True
            ),
            
            # ========== 含错别字的查询 ==========
            TestCase(
                name="错别字_摇控",
                query="摇控器怎么配对",
                category="typo",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["遥控器", "配对"],
                expected_has_filter=True
            ),
            TestCase(
                name="错别字_电池",
                query="电沲充不满",
                category="typo",
                expected_primary_intent=PrimaryIntent.TROUBLESHOOTING,
                expected_keywords_in_rewrite=["电池", "充电"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.FAULT_DIAGNOSIS
            ),
            
            # ========== 多意图查询 ==========
            TestCase(
                name="多意图_起飞和降落",
                query="怎么起飞和降落",
                category="multi_intent",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["起飞", "降落"],
                expected_has_filter=True
            ),
            TestCase(
                name="多意图_参数和校准",
                query="最大高度是多少以及怎么校准指南针",
                category="multi_intent",
                expected_primary_intent=PrimaryIntent.PARAMETER_QUERY,
                expected_keywords_in_rewrite=["高度", "指南针", "校准"],
                expected_has_filter=True
            ),
            
            # ========== 模糊查询 ==========
            TestCase(
                name="模糊_单一词汇",
                query="怎么飞",
                category="vague",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["起飞"],
                expected_has_filter=True
            ),
            TestCase(
                name="模糊_宽泛问题",
                query="电池问题",
                category="vague",
                expected_primary_intent=PrimaryIntent.TROUBLESHOOTING,
                expected_keywords_in_rewrite=["电池"],
                expected_has_filter=True
            ),
            
            # ========== 无效查询 ==========
            TestCase(
                name="无效_空查询",
                query="",
                category="invalid",
                expected_primary_intent=PrimaryIntent.UNKNOWN,
                expected_keywords_in_rewrite=[],
                expected_has_filter=False
            ),
            TestCase(
                name="无效_纯语气词",
                query="请问",
                category="invalid",
                expected_primary_intent=PrimaryIntent.UNKNOWN,
                expected_keywords_in_rewrite=[],
                expected_has_filter=False
            ),
            
            # ========== 特定场景查询 ==========
            TestCase(
                name="故障_飞丢",
                query="飞丢了怎么找回",
                category="standard",
                expected_primary_intent=PrimaryIntent.TROUBLESHOOTING,
                expected_keywords_in_rewrite=["飞行器丢失", "找回"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.LOST_DRONE
            ),
            TestCase(
                name="安全_禁飞",
                query="禁飞区有哪些",
                category="standard",
                expected_primary_intent=PrimaryIntent.SAFETY_COMPLIANCE,
                expected_keywords_in_rewrite=["禁飞区"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.NO_FLY_ZONE
            ),
            TestCase(
                name="对比_机型",
                query="Mavic 3和Mavic 4有什么区别",
                category="standard",
                expected_primary_intent=PrimaryIntent.COMPARISON,
                expected_keywords_in_rewrite=["Mavic 3", "Mavic 4"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.MODEL_COMPARISON,
                expected_models=["Mavic 3", "Mavic 4"]
            ),
            TestCase(
                name="维护_电池保养",
                query="电池怎么保养",
                category="standard",
                expected_primary_intent=PrimaryIntent.MAINTENANCE,
                expected_keywords_in_rewrite=["电池", "保养"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.BATTERY_CARE
            ),
            TestCase(
                name="固件升级",
                query="固件怎么升级",
                category="standard",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["固件", "升级"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.FIRMWARE_UPDATE
            ),
            TestCase(
                name="航线规划",
                query="怎么规划航线",
                category="standard",
                expected_primary_intent=PrimaryIntent.OPERATION_QUERY,
                expected_keywords_in_rewrite=["航线", "规划"],
                expected_has_filter=True,
                expected_secondary_intent=SecondaryIntent.ROUTE_PLANNING
            ),
        ]
        
        return test_cases
    
    def run_tests(self) -> Dict[str, Any]:
        """
        运行测试
        
        Returns:
            测试结果字典
        """
        test_cases = self.get_test_cases()
        print(f"共 {len(test_cases)} 个测试用例")
        print("=" * 80)
        
        # 处理所有查询
        contexts = []
        expected_intents = []
        expected_rewrites = []
        expected_filters = []
        
        for i, tc in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {tc.name}")
            print(f"查询: {tc.query}")
            print(f"类别: {tc.category}")
            
            # 处理查询
            context = self.system.process_query(tc.query)
            contexts.append(context)
            expected_intents.append(tc.expected_primary_intent)
            expected_rewrites.append(' '.join(tc.expected_keywords_in_rewrite))
            expected_filters.append(tc.expected_has_filter)
            
            # 记录详细结果
            result_detail = {
                "name": tc.name,
                "query": tc.query,
                "category": tc.category,
                "cleaned": context.enhanced.cleaned_query,
                "precise": context.enhanced.precise_query,
                "query_type": context.enhanced.query_type.value,
                "primary_intent": context.intent.primary_intent.value,
                "expected_intent": tc.expected_primary_intent.value,
                "intent_correct": context.intent.primary_intent == tc.expected_primary_intent,
                "intent_confidence": context.intent.confidence,
                "has_filter": context.filter_condition.chroma_filter is not None,
                "expected_filter": tc.expected_has_filter,
                "filter_correct": (context.filter_condition.chroma_filter is not None) == tc.expected_has_filter,
                "processing_time_ms": context.processing_time_ms
            }
            self.test_results.append(result_detail)
            
            print(f"  清洗: {context.enhanced.cleaned_query}")
            print(f"  精准: {context.enhanced.precise_query}")
            print(f"  意图: {context.intent.primary_intent.value} (期望: {tc.expected_primary_intent.value})")
            print(f"  匹配: {'[OK]' if result_detail['intent_correct'] else '[FAIL]'}")
        
        # 评估指标
        print("\n" + "=" * 80)
        print("评估结果")
        print("=" * 80)
        
        # 1. 意图识别准确率
        intent_eval = self.evaluator.evaluate_intent_accuracy(contexts, expected_intents)
        print(f"\n1. 意图识别准确率: {intent_eval['accuracy']:.2%}")
        print(f"   正确: {intent_eval['correct']}/{intent_eval['total']}")
        
        # 2. 查询改写准确率
        rewrite_eval = self.evaluator.evaluate_query_rewrite_accuracy(contexts, expected_rewrites)
        print(f"\n2. 查询改写准确率: {rewrite_eval['accuracy']:.2%}")
        print(f"   正确: {rewrite_eval['correct']}/{rewrite_eval['total']}")
        
        # 3. 过滤条件生成准确率
        filter_eval = self.evaluator.evaluate_filter_accuracy(contexts, expected_filters)
        print(f"\n3. 过滤条件生成准确率: {filter_eval['accuracy']:.2%}")
        print(f"   正确: {filter_eval['correct']}/{filter_eval['total']}")
        
        # 按类别统计
        category_stats = self._analyze_by_category()
        
        # 生成报告
        report = {
            "test_time": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "metrics": {
                "intent_accuracy": intent_eval,
                "rewrite_accuracy": rewrite_eval,
                "filter_accuracy": filter_eval
            },
            "category_analysis": category_stats,
            "details": self.test_results,
            "system_stats": self.system.get_stats()
        }
        
        # 保存报告
        with open("retrieval_system_test_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存: retrieval_system_test_report.json")
        
        return report
    
    def _analyze_by_category(self) -> Dict[str, Any]:
        """按类别分析测试结果"""
        categories = {}
        
        for result in self.test_results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {
                    "total": 0,
                    "intent_correct": 0,
                    "filter_correct": 0
                }
            
            categories[cat]["total"] += 1
            if result["intent_correct"]:
                categories[cat]["intent_correct"] += 1
            if result["filter_correct"]:
                categories[cat]["filter_correct"] += 1
        
        # 计算准确率
        for cat, stats in categories.items():
            stats["intent_accuracy"] = round(stats["intent_correct"] / stats["total"], 4)
            stats["filter_accuracy"] = round(stats["filter_correct"] / stats["total"], 4)
        
        return categories
    
    def print_detailed_analysis(self, report: Dict):
        """打印详细分析"""
        print("\n" + "=" * 80)
        print("各类别测试结果分析")
        print("=" * 80)
        
        for cat, stats in report["category_analysis"].items():
            print(f"\n【{cat}】")
            print(f"  测试数: {stats['total']}")
            print(f"  意图准确率: {stats['intent_accuracy']:.2%}")
            print(f"  过滤准确率: {stats['filter_accuracy']:.2%}")
        
        # 错误案例分析
        print("\n" + "=" * 80)
        print("错误案例分析")
        print("=" * 80)
        
        errors = [r for r in self.test_results if not r["intent_correct"]]
        if errors:
            for err in errors[:5]:  # 只显示前5个
                print(f"\n查询: {err['query']}")
                print(f"  类别: {err['category']}")
                print(f"  预测意图: {err['primary_intent']}")
                print(f"  期望意图: {err['expected_intent']}")
        else:
            print("\n无错误案例")


def run_quick_test():
    """快速测试单个查询"""
    system = IntelligentRetrievalSystem(use_llm_for_intent=False)
    
    print("=" * 80)
    print("智能检索系统 - 快速测试")
    print("=" * 80)
    
    while True:
        print("\n请输入查询（输入 q 退出）:")
        query = input().strip()
        
        if query.lower() == 'q':
            break
        
        if not query:
            continue
        
        result = system.process_query(query)
        print(system.explain_processing(result))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互模式
        run_quick_test()
    else:
        # 自动测试模式
        tester = RetrievalSystemTester()
        report = tester.run_tests()
        tester.print_detailed_analysis(report)
