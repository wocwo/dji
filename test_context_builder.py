"""
上下文组装器与提示词工程测试

测试覆盖：
1. Chunk预处理（过滤、意图排序）
2. 特殊内容处理（表格、步骤、警告、故障码）
3. 结构化上下文组装（6种意图策略）
4. 提示词模板渲染
5. Prompt引擎端到端
6. ResponseValidator校验
"""

import unittest
from context_builder import (
    ChunkPreprocessor,
    SpecialContentHandler,
    StructuredContextBuilder,
    PromptTemplate,
    PromptEngine,
    ResponseValidator,
    ProcessedChunk,
    build_prompt,
    build_context
)
from intent_classifier import PrimaryIntent, SecondaryIntent, IntentClassification
from reranker import RerankResult, RerankedChunk
from multi_path_retriever import RetrievalChannel


class MockDataFactory:
    """测试数据工厂"""
    
    @staticmethod
    def make_intent(primary: PrimaryIntent, secondary: SecondaryIntent) -> IntentClassification:
        return IntentClassification(
            primary_intent=primary,
            secondary_intent=secondary,
            confidence=0.9,
            detected_keywords=["测试"],
            reasoning="测试用"
        )
    
    @staticmethod
    def make_reranked_chunk(
        chunk_id: str,
        document: str,
        final_score: float,
        metadata: dict = None
    ) -> RerankedChunk:
        return RerankedChunk(
            chunk_id=chunk_id,
            document=document,
            metadata=metadata or {},
            rerank_score=final_score,
            rrf_score=final_score,
            final_score=final_score,
            sources=[RetrievalChannel.SEMANTIC],
            raw_score=final_score,
            passed_threshold=final_score >= 0.3
        )
    
    @staticmethod
    def make_rerank_result(chunks: list) -> RerankResult:
        passed = [c for c in chunks if c.passed_threshold]
        return RerankResult(
            query="测试查询",
            reranked_chunks=chunks,
            input_count=len(chunks),
            output_count=len(passed),
            filtered_count=len(chunks) - len(passed),
            avg_score=0.5,
            max_score=1.0,
            min_score=0.0,
            rerank_time_ms=10.0,
            threshold=0.3
        )


class TestChunkPreprocessor(unittest.TestCase):
    """测试Chunk预处理器"""
    
    def test_filter_low_score_chunks(self):
        """测试过滤低于阈值的Chunk"""
        chunks = [
            MockDataFactory.make_reranked_chunk("c1", "内容1", 0.5),
            MockDataFactory.make_reranked_chunk("c2", "内容2", 0.2),
            MockDataFactory.make_reranked_chunk("c3", "内容3", 0.1),
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
        
        preprocessor = ChunkPreprocessor(score_threshold=0.3)
        processed = preprocessor.process(result, intent)
        
        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0].chunk_id, "c1")
    
    def test_sort_operation_by_section_order(self):
        """测试操作步骤查询按章节顺序排序"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "1. 打开电源", 0.5, {"section": "3 起飞准备", "type": "操作"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "1. 安装桨叶", 0.6, {"section": "1 准备工作", "type": "操作"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c3", "1. 校准指南针", 0.4, {"section": "2 校准操作", "type": "操作"}
            ),
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.OPERATION_QUERY, SecondaryIntent.TAKEOFF_LANDING)
        
        preprocessor = ChunkPreprocessor(score_threshold=0.3)
        processed = preprocessor.process(result, intent)
        
        section_orders = [p.section_order for p in processed]
        self.assertEqual(section_orders, [1, 2, 3])
    
    def test_sort_parameter_table_first(self):
        """测试参数查询表格类优先"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "飞行高度为1000米", 0.5, {"type": "参数"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "| 参数 | 值 |\n|---|---|\n| 高度 | 1000m |", 0.45, {"type": "参数"}
            ),
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.PARAMETER_QUERY, SecondaryIntent.FLIGHT_PARAM)
        
        preprocessor = ChunkPreprocessor(score_threshold=0.3)
        processed = preprocessor.process(result, intent)
        
        self.assertEqual(processed[0].content_type, "table")
    
    def test_sort_safety_warning_first(self):
        """测试安全查询警告类优先"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "飞行前请检查电池电量", 0.5, {"type": "安全"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "警告：禁止在人群密集区域飞行", 0.4, {"type": "安全"}
            ),
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.SAFETY_COMPLIANCE, SecondaryIntent.FLIGHT_GUIDE)
        
        preprocessor = ChunkPreprocessor(score_threshold=0.3)
        processed = preprocessor.process(result, intent)
        
        self.assertEqual(processed[0].warning_level, "critical")


class TestSpecialContentHandler(unittest.TestCase):
    """测试特殊内容处理器"""
    
    def setUp(self):
        self.handler = SpecialContentHandler()
        self.intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
    
    def _make_chunk(self, content: str, ctype: str = "text", level: str = "none") -> ProcessedChunk:
        return ProcessedChunk(
            chunk_id="t1",
            content=content,
            metadata={},
            rerank_score=0.5,
            rrf_score=0.5,
            final_score=0.5,
            sources=["semantic"],
            content_type=ctype,
            warning_level=level
        )
    
    def test_process_table(self):
        """测试表格保留格式"""
        content = "参数说明\n| 参数 | 值 |\n|---|---|\n| 高度 | 1000m |\n其他说明"
        chunk = self._make_chunk(content, "table")
        result = self.handler.process_chunk(chunk, self.intent)
        self.assertIn("| 参数 | 值 |", result)
        self.assertIn("| 高度 | 1000m |", result)
    
    def test_process_steps(self):
        """测试步骤编号规范化"""
        content = "（1）打开电源\n2、连接遥控器\n步骤3：启动电机"
        chunk = self._make_chunk(content, "step")
        result = self.handler.process_chunk(chunk, self.intent)
        self.assertIn("1. 打开电源", result)
        self.assertIn("2. 连接遥控器", result)
        self.assertIn("3. 启动电机", result)
    
    def test_process_warning(self):
        """测试警告前缀"""
        content = "禁止在雨天飞行"
        chunk = self._make_chunk(content, "warning", "critical")
        result = self.handler.process_chunk(chunk, self.intent)
        self.assertTrue(result.startswith("【⚠️ 严重警告】"))
    
    def test_process_error_code(self):
        """测试故障码格式化"""
        content = "故障码 30045\n原因：IMU异常\n解决方法：重启飞行器"
        chunk = self._make_chunk(content, "error_code")
        result = self.handler.process_chunk(chunk, self.intent)
        self.assertIn("故障码：30045", result)
        self.assertIn("原因：IMU异常", result)
        self.assertIn("解决方法：重启飞行器", result)


class TestStructuredContextBuilder(unittest.TestCase):
    """测试结构化上下文组装器"""
    
    def _build_result(self, chunks):
        return MockDataFactory.make_rerank_result(chunks)
    
    def test_parameter_context_prioritizes_tables(self):
        """测试参数查询上下文优先展示表格"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "Mavic 4 Pro 飞行高度为1000米", 0.5,
                {"model": "Mavic 4 Pro", "section": "飞行参数", "type": "参数"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "| 参数 | 值 |\n|---|---|\n| 高度 | 1000m |", 0.45,
                {"model": "Mavic 4 Pro", "section": "技术规格", "type": "参数"}
            ),
        ]
        intent = MockDataFactory.make_intent(PrimaryIntent.PARAMETER_QUERY, SecondaryIntent.FLIGHT_PARAM)
        builder = StructuredContextBuilder()
        ctx = builder.build(self._build_result(chunks), intent, "Mavic 4 Pro")
        
        self.assertIn("参数规格表", ctx.markdown_text)
        self.assertIn("参数说明", ctx.markdown_text)
        # 表格应该出现在说明之前
        table_idx = ctx.markdown_text.find("参数规格表")
        desc_idx = ctx.markdown_text.find("参数说明")
        self.assertLess(table_idx, desc_idx)
    
    def test_operation_context_merges_steps(self):
        """测试操作步骤查询合并连续步骤"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "1. 打开遥控器\n2. 连接手机", 0.5,
                {"section": "1 准备", "type": "步骤"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "1. 展开桨叶\n2. 检查电量", 0.5,
                {"section": "2 检查", "type": "步骤"}
            ),
        ]
        intent = MockDataFactory.make_intent(PrimaryIntent.OPERATION_QUERY, SecondaryIntent.TAKEOFF_LANDING)
        builder = StructuredContextBuilder()
        ctx = builder.build(self._build_result(chunks), intent)
        
        # 步骤应该被重新编号为 1-4
        self.assertIn("1. 打开遥控器", ctx.markdown_text)
        self.assertIn("2. 连接手机", ctx.markdown_text)
        self.assertIn("3. 展开桨叶", ctx.markdown_text)
        self.assertIn("4. 检查电量", ctx.markdown_text)
    
    def test_troubleshooting_context_groups_error_codes(self):
        """测试故障排查查询按故障码分组"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "故障码 30045\n原因：IMU异常", 0.5, {}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "故障码 30045\n解决方法：重启", 0.5, {}
            ),
            MockDataFactory.make_reranked_chunk(
                "c3", "故障码 30046\n原因：指南针异常", 0.5, {}
            ),
        ]
        intent = MockDataFactory.make_intent(PrimaryIntent.TROUBLESHOOTING, SecondaryIntent.ERROR_CODE)
        builder = StructuredContextBuilder()
        ctx = builder.build(self._build_result(chunks), intent)
        
        self.assertIn("故障码：30045", ctx.markdown_text)
        self.assertIn("故障码：30046", ctx.markdown_text)
    
    def test_safety_context_warning_first(self):
        """测试安全查询警告置顶"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "飞行前请检查环境", 0.5, {"type": "安全"}
            ),
            MockDataFactory.make_reranked_chunk(
                "c2", "警告：禁飞区禁止起飞", 0.5, {"type": "安全"}
            ),
        ]
        intent = MockDataFactory.make_intent(PrimaryIntent.SAFETY_COMPLIANCE, SecondaryIntent.NO_FLY_ZONE)
        builder = StructuredContextBuilder()
        ctx = builder.build(self._build_result(chunks), intent)
        
        self.assertIn("安全警告", ctx.markdown_text)
        self.assertIn("安全规范说明", ctx.markdown_text)
        warn_idx = ctx.markdown_text.find("安全警告")
        desc_idx = ctx.markdown_text.find("安全规范说明")
        self.assertLess(warn_idx, desc_idx)
    
    def test_global_header(self):
        """测试全局信息头部"""
        chunks = [
            MockDataFactory.make_reranked_chunk("c1", "测试内容", 0.5, {})
        ]
        intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
        builder = StructuredContextBuilder()
        ctx = builder.build(self._build_result(chunks), intent, "Mavic 4 Pro")
        
        self.assertIn("**查询机型**：Mavic 4 Pro", ctx.markdown_text)
        self.assertIn("**查询意图**：未知意图 → 未知", ctx.markdown_text)
        self.assertIn("以下内容全部来自大疆官方产品说明书", ctx.markdown_text)


class TestPromptTemplate(unittest.TestCase):
    """测试提示词模板"""
    
    def test_base_template_contains_hallucination_control(self):
        """测试基础模板包含幻觉防控约束"""
        intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="测试上下文", chunk_count=1, total_chars=10, assembly_time_ms=1.0)
        prompt = PromptTemplate.render("测试问题", ctx, intent)
        
        self.assertIn("以下内容全部来自大疆官方产品说明书", prompt)
        self.assertIn("信息来源约束", prompt)
        self.assertIn("严禁使用以下模糊表述", prompt)
        self.assertIn("测试问题", prompt)
    
    def test_parameter_template(self):
        """测试参数查询模板"""
        intent = MockDataFactory.make_intent(PrimaryIntent.PARAMETER_QUERY, SecondaryIntent.FLIGHT_PARAM)
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="测试", chunk_count=1, total_chars=4, assembly_time_ms=1.0)
        prompt = PromptTemplate.render("最大飞行高度是多少", ctx, intent)
        
        self.assertIn("参数/规格查询", prompt)
        self.assertIn("优先以表格或列表形式呈现参数", prompt)
    
    def test_safety_template(self):
        """测试安全合规模板"""
        intent = MockDataFactory.make_intent(PrimaryIntent.SAFETY_COMPLIANCE, SecondaryIntent.NO_FLY_ZONE)
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="测试", chunk_count=1, total_chars=4, assembly_time_ms=1.0)
        prompt = PromptTemplate.render("禁飞区有哪些", ctx, intent)
        
        self.assertIn("安全合规查询", prompt)
        self.assertIn("不得弱化风险描述", prompt)


class TestPromptEngine(unittest.TestCase):
    """测试Prompt引擎"""
    
    def test_build_prompt_end_to_end(self):
        """测试端到端Prompt构建"""
        chunks = [
            MockDataFactory.make_reranked_chunk(
                "c1", "Mavic 4 Pro 最大飞行高度为 1000 米", 0.5,
                {"model": "Mavic 4 Pro", "section": "飞行参数", "type": "参数", "keywords": ["高度", "飞行"]}
            )
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.PARAMETER_QUERY, SecondaryIntent.FLIGHT_PARAM)
        
        engine = PromptEngine()
        prompt_result = engine.build_prompt("Mavic 4 Pro最大飞行高度", result, intent, "Mavic 4 Pro")
        
        self.assertIn("prompt", prompt_result)
        self.assertIn("context", prompt_result)
        self.assertIn("metadata", prompt_result)
        self.assertGreater(prompt_result["metadata"]["prompt_chars"], 0)
        self.assertEqual(prompt_result["metadata"]["chunk_count"], 1)


class TestResponseValidator(unittest.TestCase):
    """测试回答校验器"""
    
    def test_detects_forbidden_phrases(self):
        """测试检测模糊表述"""
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="测试", chunk_count=1, total_chars=4, assembly_time_ms=1.0)
        result = ResponseValidator.validate("我觉得可能是1000米", ctx)
        
        self.assertTrue(any(i["type"] == "forbidden_phrase" for i in result["issues"]))
    
    def test_missing_no_info_reply(self):
        """测试检测无信息标准回答缺失"""
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="", chunk_count=0, total_chars=0, assembly_time_ms=1.0)
        result = ResponseValidator.validate("根据我的知识，应该是这样的", ctx)
        
        self.assertTrue(any(i["type"] == "missing_no_info_reply" for i in result["issues"]))
        self.assertFalse(result["passed"])
    
    def test_passes_clean_response(self):
        """测试正常回答通过校验"""
        from context_builder import AssembledContext
        ctx = AssembledContext(markdown_text="高度为 1000 米", chunk_count=1, total_chars=10, assembly_time_ms=1.0)
        result = ResponseValidator.validate("最大飞行高度为 1000 米（来源：Mavic 4 Pro / 飞行参数）。", ctx)
        
        self.assertTrue(result["passed"])


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def test_build_context(self):
        """测试build_context便捷函数"""
        chunks = [
            MockDataFactory.make_reranked_chunk("c1", "测试", 0.5, {})
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
        
        ctx = build_context(result, intent, target_model="Test")
        self.assertEqual(ctx.chunk_count, 1)
    
    def test_build_prompt(self):
        """测试build_prompt便捷函数"""
        chunks = [
            MockDataFactory.make_reranked_chunk("c1", "测试", 0.5, {})
        ]
        result = MockDataFactory.make_rerank_result(chunks)
        intent = MockDataFactory.make_intent(PrimaryIntent.UNKNOWN, SecondaryIntent.UNKNOWN)
        
        pr = build_prompt("问题", result, intent)
        self.assertIn("prompt", pr)
        self.assertIn("context", pr)


if __name__ == "__main__":
    unittest.main()
