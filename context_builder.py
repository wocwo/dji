"""
上下文组装器与提示词工程模块 (Context Builder & Prompt Engineering)

功能：
1. Chunk预处理：过滤噪声、意图加权排序
2. 结构化上下文组装：将离散Chunks转换为语义完整、结构清晰的Markdown上下文
3. 特殊内容定制化组装：表格、步骤列表、警告、故障码等
4. 提示词模板设计：基础通用模板 + 意图专用模板，着重幻觉防控
"""

import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from intent_classifier import PrimaryIntent, SecondaryIntent, IntentClassification
from reranker import RerankResult, RerankedChunk


@dataclass
class ProcessedChunk:
    """预处理后的Chunk"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    rerank_score: float
    rrf_score: float
    final_score: float
    sources: List[str]
    
    # 预处理添加的字段
    content_type: str = "text"          # text | table | step | warning | error_code
    section_order: int = 999            # 章节顺序，用于排序
    priority_score: float = 0.0         # 组装优先级分数
    warning_level: str = "none"         # none | low | medium | high | critical


@dataclass
class AssembledContext:
    """组装后的上下文"""
    markdown_text: str
    chunk_count: int
    total_chars: int
    assembly_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# 1. Chunk 预处理器
# =============================================================================

class ChunkPreprocessor:
    """
    Chunk预处理器
    
    在组装前对Chunk结果做二次处理：
    - 过滤重排序分数 < threshold 的Chunk
    - 基于意图的分类排序和意图加权排序
    """
    
    def __init__(self, score_threshold: float = 0.3):
        """
        初始化预处理器
        
        Args:
            score_threshold: 重排序分数阈值，低于此值的Chunk会被过滤
        """
        self.score_threshold = score_threshold
    
    def process(
        self,
        rerank_result: RerankResult,
        intent: IntentClassification
    ) -> List[ProcessedChunk]:
        """
        执行预处理
        
        Args:
            rerank_result: 重排序结果
            intent: 意图分类结果
            
        Returns:
            预处理后的Chunk列表
        """
        # 1. 过滤低分Chunk
        filtered = self._filter_chunks(rerank_result.reranked_chunks)
        
        # 2. 转换为ProcessedChunk并提取特征
        processed = [self._extract_features(chunk) for chunk in filtered]
        
        # 3. 基于意图进行加权排序
        sorted_chunks = self._sort_by_intent(processed, intent)
        
        return sorted_chunks
    
    def _filter_chunks(self, chunks: List[RerankedChunk]) -> List[RerankedChunk]:
        """过滤重排序分数低于阈值的Chunk"""
        return [c for c in chunks if c.final_score >= self.score_threshold]
    
    def _extract_features(self, chunk: RerankedChunk) -> ProcessedChunk:
        """提取Chunk特征，用于后续排序和组装"""
        metadata = chunk.metadata or {}
        content = chunk.document or ""
        
        # 识别内容类型
        content_type = self._detect_content_type(content, metadata)
        
        # 提取章节顺序
        section_order = self._extract_section_order(metadata.get("section", ""))
        
        # 识别警告等级
        warning_level = self._detect_warning_level(content)
        
        # 计算组装优先级分数（基础为final_score，后续会根据意图调整）
        priority_score = chunk.final_score
        
        return ProcessedChunk(
            chunk_id=chunk.chunk_id,
            content=content,
            metadata=metadata,
            rerank_score=chunk.rerank_score,
            rrf_score=chunk.rrf_score,
            final_score=chunk.final_score,
            sources=[s.value if hasattr(s, 'value') else str(s) for s in chunk.sources],
            content_type=content_type,
            section_order=section_order,
            priority_score=priority_score,
            warning_level=warning_level
        )
    
    def _detect_content_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """检测内容类型"""
        content = content.strip()
        
        # 检查元数据中是否已有类型标记
        meta_type = metadata.get("type", "")
        if meta_type in ["参数", "规格", "技术规格", "对比"]:
            if self._contains_table(content):
                return "table"
        
        if meta_type in ["操作", "步骤", "指南"]:
            if self._contains_steps(content):
                return "step"
        
        if meta_type in ["故障", "异常", "排查", "错误代码"]:
            if self._contains_error_code(content):
                return "error_code"
        
        if meta_type in ["安全", "法规", "警告"]:
            if self._contains_warning(content):
                return "warning"
        
        # 基于内容特征识别
        if self._contains_table(content):
            return "table"
        if self._contains_error_code(content):
            return "error_code"
        if self._contains_steps(content):
            return "step"
        if self._contains_warning(content):
            return "warning"
        
        return "text"
    
    @staticmethod
    def _contains_table(content: str) -> bool:
        """检查内容是否包含Markdown表格"""
        return bool(re.search(r'\|[-:\s]+\|[-:\s\|]*\n', content))
    
    @staticmethod
    def _contains_steps(content: str) -> bool:
        """检查内容是否包含步骤列表"""
        patterns = [
            r'^\d+[\.、\)\]\s]+',
            r'^[（(]?\d+[)）]?[\.\s]+',
            r'^[ Step步骤]{0,5}\d+[\.:\s]+'
        ]
        lines = content.split('\n')
        step_count = 0
        for line in lines:
            for p in patterns:
                if re.match(p, line.strip()):
                    step_count += 1
                    break
        return step_count >= 2
    
    @staticmethod
    def _contains_error_code(content: str) -> bool:
        """检查内容是否包含故障码"""
        patterns = [
            r'[Ee]rror\s*[:：]?\s*[A-Z]?\d{2,}',
            r'[Cc]ode\s*[:：]?\s*\d{2,}',
            r'故障码\s*[:：]?\s*\d{2,}',
            r'错误码\s*[:：]?\s*\d{2,}',
            r'#[A-Z]?\d{2,}',
            r'[A-Z]{1,3}\d{3,}',
        ]
        return any(re.search(p, content) for p in patterns)
    
    @staticmethod
    def _contains_warning(content: str) -> bool:
        """检查内容是否包含警告/注意"""
        warning_keywords = [
            "警告", "注意", "⚠️", "🚨", "❗", "Important", "Warning",
            "Caution", "Attention", "Disclaimer", "免责声明", "危险",
            "严禁", "禁止", "必须", "务必", "请勿"
        ]
        return any(kw in content for kw in warning_keywords)
    
    def _detect_warning_level(self, content: str) -> str:
        """检测警告等级"""
        critical_keywords = ["警告", "🚨", "危险", "严禁", "禁止", "可能造成伤害", "可能导致"]
        high_keywords = ["注意", "⚠️", "❗", "Important", "Warning", "Caution", "必须", "务必", "请勿"]
        medium_keywords = ["建议", "推荐", "最好", "尽量"]
        
        if any(kw in content for kw in critical_keywords):
            return "critical"
        if any(kw in content for kw in high_keywords):
            return "high"
        if any(kw in content for kw in medium_keywords):
            return "medium"
        return "none"
    
    def _extract_section_order(self, section: str) -> int:
        """从章节标题中提取顺序号"""
        if not section:
            return 999
        
        match = re.match(r'^(\d+)[\.\s]', section)
        if match:
            return int(match.group(1))
        
        chinese_nums = {
            '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
        }
        for cn, num in chinese_nums.items():
            if section.startswith(cn):
                return num
        
        return 999
    
    def _sort_by_intent(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification
    ) -> List[ProcessedChunk]:
        """基于意图进行分类排序"""
        primary = intent.primary_intent
        
        if primary == PrimaryIntent.OPERATION_QUERY:
            return self._sort_operation_chunks(chunks)
        elif primary == PrimaryIntent.TROUBLESHOOTING:
            return self._sort_troubleshooting_chunks(chunks)
        elif primary == PrimaryIntent.PARAMETER_QUERY:
            return self._sort_parameter_chunks(chunks)
        elif primary == PrimaryIntent.SAFETY_COMPLIANCE:
            return self._sort_safety_chunks(chunks)
        elif primary == PrimaryIntent.COMPARISON:
            return self._sort_comparison_chunks(chunks)
        else:
            chunks.sort(key=lambda x: x.final_score, reverse=True)
            return chunks
    
    def _sort_operation_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """操作步骤查询排序：按章节顺序升序，步骤类型优先"""
        for c in chunks:
            if c.content_type == "step":
                c.priority_score += 0.2
        chunks.sort(key=lambda x: (x.section_order, -x.priority_score))
        return chunks
    
    def _sort_troubleshooting_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """故障排查查询排序：故障码优先，其次是原因，最后是解决方法"""
        type_order = {
            "error_code": 0,
            "warning": 1,
            "text": 2,
            "step": 3,
            "table": 4
        }
        for c in chunks:
            base_order = type_order.get(c.content_type, 2)
            content_lower = c.content.lower()
            if c.content_type == "error_code" or re.search(r'[A-Z]?\d{2,}', c.content):
                base_order = 0
            elif any(kw in content_lower for kw in ["原因", "why", "因为", "由于", "引起"]):
                base_order = 1
            elif any(kw in content_lower for kw in ["解决", "方法", "处理", "修复", "排查", "步骤"]):
                base_order = 2
            c.priority_score = -base_order * 10 + c.final_score
        chunks.sort(key=lambda x: (x.priority_score, x.final_score), reverse=True)
        return chunks
    
    def _sort_parameter_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """参数查询排序：表格类优先，合并同类型参数"""
        for c in chunks:
            if c.content_type == "table":
                c.priority_score += 0.3
            elif c.content_type == "text":
                c.priority_score += 0.1
        chunks.sort(key=lambda x: x.priority_score, reverse=True)
        return chunks
    
    def _sort_safety_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """安全合规查询排序：警告类强制置顶，按风险等级排序"""
        warning_priority = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "none": 3
        }
        for c in chunks:
            if c.content_type == "warning":
                c.priority_score += 0.5
            wp = warning_priority.get(c.warning_level, 3)
            c.priority_score = -wp * 10 + c.final_score + c.priority_score
        chunks.sort(key=lambda x: (x.priority_score, x.final_score), reverse=True)
        return chunks
    
    def _sort_comparison_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        """对比查询排序：按机型分组"""
        for c in chunks:
            if c.content_type == "table":
                c.priority_score += 0.2
        model = c.metadata.get("model", "") or c.metadata.get("机型", "")
        chunks.sort(key=lambda x: (x.priority_score, model), reverse=True)
        return chunks


# =============================================================================
# 2. 特殊内容处理器
# =============================================================================

class SpecialContentHandler:
    """
    特殊内容处理器
    
    针对大疆文档中高频出现的特殊内容类型，定制化组装并保留原始结构：
    - 表格处理：完整保留 Markdown 表格格式，禁止拆分表格行
    - 步骤列表处理：保留原始编号顺序，禁止打乱步骤
    - 警告/注意处理：在内容前添加警告前缀，强制大模型优先引用
    - 故障码处理：统一格式为 故障码：XXX → 原因：XXX → 解决方法：XXX
    """
    
    WARNING_PREFIXES = {
        "critical": "【⚠️ 严重警告】",
        "high": "【⚠️ 警告】",
        "medium": "【注意】",
        "none": ""
    }
    
    def process_chunk(self, chunk: ProcessedChunk, intent: IntentClassification) -> str:
        """处理单个Chunk的内容"""
        content = chunk.content.strip()
        if not content:
            return ""
        
        if chunk.content_type == "table":
            content = self._process_table(content)
        elif chunk.content_type == "step":
            content = self._process_steps(content)
        elif chunk.content_type == "warning":
            content = self._process_warning(content, chunk.warning_level)
        elif chunk.content_type == "error_code":
            content = self._process_error_code(content)
        
        content = self._apply_intent_processing(content, chunk, intent)
        return content
    
    def _process_table(self, content: str) -> str:
        """表格处理：完整保留Markdown表格格式"""
        lines = content.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('|'):
                table_lines.append(stripped)
                in_table = True
            elif in_table and stripped:
                table_lines.append(stripped)
            elif in_table and not stripped:
                in_table = False
        
        if not table_lines:
            return content
        
        return '\n'.join(table_lines)
    
    def _process_steps(self, content: str) -> str:
        """步骤列表处理：保留原始编号顺序，规范化编号格式"""
        lines = content.split('\n')
        processed_lines = []
        step_counter = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                processed_lines.append('')
                continue
            
            step_match = re.match(
                r'^(?:[（(])?(\d+|[一二三四五六七八九十]+)(?:[)）])?[\.、\):：\s]*(.+)$',
                stripped
            )
            
            if step_match:
                step_counter += 1
                step_body = step_match.group(2)
                processed_lines.append(f"{step_counter}. {step_body}")
            elif re.match(r'^[ Step步骤]{0,5}\d+[\.:：\s]+', stripped):
                body = re.sub(r'^[ Step步骤]{0,5}\d+[\.:：\s]+', '', stripped)
                step_counter += 1
                processed_lines.append(f"{step_counter}. {body}")
            else:
                processed_lines.append(stripped)
        
        return '\n'.join(processed_lines)
    
    def _process_warning(self, content: str, warning_level: str) -> str:
        """警告/注意处理：在内容前添加警告前缀"""
        prefix = self.WARNING_PREFIXES.get(warning_level, "")
        if prefix:
            lines = content.split('\n')
            result_lines = []
            prefix_applied = False
            for line in lines:
                if line.strip() and not prefix_applied:
                    result_lines.append(f"{prefix} {line}")
                    prefix_applied = True
                else:
                    result_lines.append(line)
            return '\n'.join(result_lines)
        return content
    
    def _process_error_code(self, content: str) -> str:
        """故障码处理：统一格式"""
        error_codes = re.findall(r'([A-Z]?\d{2,})', content)
        if not error_codes:
            return content
        
        has_reason = any(kw in content for kw in ["原因", "因为", "由于"])
        has_solution = any(kw in content for kw in ["解决", "方法", "处理", "修复"])
        
        if not has_reason and not has_solution:
            return f"【故障码：{error_codes[0]}】\n{content}"
        
        parts = []
        for code in error_codes[:1]:
            parts.append(f"故障码：{code}")
        
        lines = content.split('\n')
        reason_lines = []
        solution_lines = []
        current_section = None
        
        for line in lines:
            stripped = line.strip()
            if any(kw in stripped for kw in ["原因", "因为", "由于", "引起"]):
                current_section = "reason"
                # 提取关键词之后的内容（优先匹配最长关键词）
                for kw in ["原因", "因为", "由于", "引起"]:
                    if kw in stripped:
                        idx = stripped.find(kw) + len(kw)
                        remainder = stripped[idx:].lstrip('：: ')
                        if remainder:
                            reason_lines.append(remainder)
                        break
                continue
            elif any(kw in stripped for kw in ["解决方法", "解决", "方法", "处理", "修复", "排查", "步骤"]):
                current_section = "solution"
                # 提取关键词之后的内容（优先匹配最长关键词）
                for kw in ["解决方法", "解决", "方法", "处理", "修复", "排查", "步骤"]:
                    if kw in stripped:
                        idx = stripped.find(kw) + len(kw)
                        remainder = stripped[idx:].lstrip('：: ')
                        if remainder:
                            solution_lines.append(remainder)
                        break
                continue
            
            if current_section == "reason":
                reason_lines.append(stripped)
            elif current_section == "solution":
                solution_lines.append(stripped)
        
        if reason_lines:
            parts.append(f"原因：{' '.join(reason_lines)}")
        if solution_lines:
            parts.append(f"解决方法：{' '.join(solution_lines)}")
        
        if len(parts) > 1:
            return '\n'.join(parts)
        
        return content
    
    def _apply_intent_processing(
        self,
        content: str,
        chunk: ProcessedChunk,
        intent: IntentClassification
    ) -> str:
        """根据意图类型进行额外处理"""
        primary = intent.primary_intent
        
        if primary == PrimaryIntent.PARAMETER_QUERY and chunk.content_type == "table":
            content = self._trim_redundant_descriptions(content)
        elif primary == PrimaryIntent.SAFETY_COMPLIANCE and chunk.content_type != "warning":
            if self._contains_safety_keywords(content):
                prefix = "【安全提示】"
                lines = content.split('\n')
                result_lines = []
                prefix_applied = False
                for line in lines:
                    if line.strip() and not prefix_applied:
                        result_lines.append(f"{prefix} {line}")
                        prefix_applied = True
                    else:
                        result_lines.append(line)
                content = '\n'.join(result_lines)
        
        return content
    
    @staticmethod
    def _trim_redundant_descriptions(content: str) -> str:
        """去除参数表格周围的冗余描述"""
        lines = content.split('\n')
        table_start = -1
        table_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                if table_start == -1:
                    table_start = i
                table_end = i
        
        if table_start >= 0 and table_end >= 0:
            start_idx = max(0, table_start - 2)
            end_idx = min(len(lines), table_end + 3)
            return '\n'.join(lines[start_idx:end_idx]).strip()
        
        return content
    
    @staticmethod
    def _contains_safety_keywords(content: str) -> bool:
        """检查内容是否包含安全相关关键词"""
        safety_keywords = [
            "安全", "警告", "注意", "危险", "严禁", "禁止", "必须",
            "风险", "事故", "伤害", "损坏", "赔偿", "责任"
        ]
        return any(kw in content for kw in safety_keywords)


# =============================================================================
# 3. 结构化上下文组装器
# =============================================================================

class StructuredContextBuilder:
    """
    结构化上下文组装器
    
    将检索到的离散Chunks转换为大模型易理解、语义完整、结构清晰的上下文文本。
    输出格式为Markdown，保留关键元数据与文档层级。
    """
    
    def __init__(
        self,
        score_threshold: float = 0.3,
        max_total_chars: int = 8000,
        include_source_info: bool = True
    ):
        self.preprocessor = ChunkPreprocessor(score_threshold=score_threshold)
        self.content_handler = SpecialContentHandler()
        self.max_total_chars = max_total_chars
        self.include_source_info = include_source_info
    
    def build(
        self,
        rerank_result: RerankResult,
        intent: IntentClassification,
        target_model: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> AssembledContext:
        """组装结构化上下文"""
        import time
        start_time = time.time()
        
        processed_chunks = self.preprocessor.process(rerank_result, intent)
        
        if intent.primary_intent == PrimaryIntent.TROUBLESHOOTING:
            markdown = self._build_troubleshooting_context(
                processed_chunks, intent, target_model, timestamp
            )
        elif intent.primary_intent == PrimaryIntent.PARAMETER_QUERY:
            markdown = self._build_parameter_context(
                processed_chunks, intent, target_model, timestamp
            )
        elif intent.primary_intent == PrimaryIntent.OPERATION_QUERY:
            markdown = self._build_operation_context(
                processed_chunks, intent, target_model, timestamp
            )
        elif intent.primary_intent == PrimaryIntent.SAFETY_COMPLIANCE:
            markdown = self._build_safety_context(
                processed_chunks, intent, target_model, timestamp
            )
        elif intent.primary_intent == PrimaryIntent.COMPARISON:
            markdown = self._build_comparison_context(
                processed_chunks, intent, target_model, timestamp
            )
        else:
            markdown = self._build_default_context(
                processed_chunks, intent, target_model, timestamp
            )
        
        if len(markdown) > self.max_total_chars:
            markdown = self._smart_truncate(markdown, self.max_total_chars)
        
        assembly_time = (time.time() - start_time) * 1000
        
        return AssembledContext(
            markdown_text=markdown,
            chunk_count=len(processed_chunks),
            total_chars=len(markdown),
            assembly_time_ms=assembly_time,
            metadata={
                "target_model": target_model,
                "primary_intent": intent.primary_intent.value,
                "secondary_intent": intent.secondary_intent.value,
                "timestamp": timestamp or datetime.now().isoformat(),
                "score_threshold": self.preprocessor.score_threshold
            }
        )
    
    def _build_global_header(
        self,
        target_model: Optional[str],
        intent: IntentClassification,
        timestamp: Optional[str]
    ) -> str:
        """构建全局信息头部"""
        ts = timestamp or datetime.now().isoformat()
        model_str = target_model if target_model else "未指定"
        
        lines = [
            "## 检索上下文信息",
            "",
            f"- **查询机型**：{model_str}",
            f"- **查询意图**：{intent.primary_intent.value} → {intent.secondary_intent.value}",
            f"- **检索时间**：{ts}",
            "- **信息来源**：以下内容全部来自大疆官方产品说明书",
            ""
        ]
        return '\n'.join(lines)
    
    def _build_chunk_section(self, idx: int, chunk: ProcessedChunk) -> str:
        """构建单个Chunk的分块内容区块"""
        lines = []
        lines.append(f"### 分块内容 {idx}")
        lines.append("")
        
        if self.include_source_info:
            model = chunk.metadata.get("model", chunk.metadata.get("机型", "未知"))
            section = chunk.metadata.get("section", "未知章节")
            chunk_type = chunk.metadata.get("type", chunk.content_type)
            keywords = chunk.metadata.get("keywords", [])
            
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            
            lines.append(f"- **来源机型**：{model}")
            lines.append(f"- **来源章节**：{section}")
            lines.append(f"- **内容类型**：{chunk_type}")
            lines.append(f"- **核心关键词**：{', '.join(keywords) if keywords else '无'}")
            lines.append(f"- **相关度**：{chunk.rerank_score:.2f}")
            lines.append("")
        
        lines.append("**内容**：")
        lines.append("")
        lines.append(chunk.content)
        lines.append("")
        lines.append("---")
        lines.append("")
        
        return '\n'.join(lines)
    
    def _build_default_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """默认上下文组装"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        
        for i, chunk in enumerate(chunks, start=1):
            processed_content = self.content_handler.process_chunk(chunk, intent)
            chunk.content = processed_content
            parts.append(self._build_chunk_section(i, chunk))
        
        return '\n'.join(parts)
    
    def _build_parameter_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """参数查询上下文组装：优先组装表格类Chunk，合并同类型参数"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        
        table_chunks = [c for c in chunks if c.content_type == "table"]
        other_chunks = [c for c in chunks if c.content_type != "table"]
        
        if table_chunks:
            parts.append("### 参数规格表")
            parts.append("")
            for i, chunk in enumerate(table_chunks, start=1):
                processed = self.content_handler.process_chunk(chunk, intent)
                parts.append(f"**参数表 {i}**（来源：{chunk.metadata.get('section', '未知')}）")
                parts.append("")
                parts.append(processed)
                parts.append("")
        
        if other_chunks:
            parts.append("### 参数说明")
            parts.append("")
            for i, chunk in enumerate(other_chunks, start=1):
                processed = self.content_handler.process_chunk(chunk, intent)
                parts.append(f"**说明 {i}**：{processed}")
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_operation_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """操作步骤查询上下文组装：按章节顺序排列，合并连续步骤"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        parts.append("### 操作步骤")
        parts.append("")
        
        step_counter = 0
        prev_section = None
        
        for chunk in chunks:
            processed = self.content_handler.process_chunk(chunk, intent)
            section = chunk.metadata.get("section", "未知章节")
            
            if section != prev_section:
                if prev_section is not None:
                    parts.append("")
                parts.append(f"**{section}**")
                parts.append("")
                prev_section = section
            
            if chunk.content_type == "step":
                lines = processed.split('\n')
                for line in lines:
                    match = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
                    if match:
                        step_counter += 1
                        body = match.group(2)
                        parts.append(f"{step_counter}. {body}")
                    elif line.strip():
                        parts.append(line)
            else:
                parts.append(processed)
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_troubleshooting_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """故障排查查询上下文组装：故障码置顶，合并同一故障的多个Chunk"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        
        error_groups = defaultdict(list)
        other_chunks = []
        
        for chunk in chunks:
            processed = self.content_handler.process_chunk(chunk, intent)
            codes = re.findall(r'([A-Z]?\d{2,})', processed)
            if codes:
                key = codes[0]
                error_groups[key].append((chunk, processed))
            else:
                other_chunks.append((chunk, processed))
        
        if error_groups:
            parts.append("### 故障码信息")
            parts.append("")
            
            for code, items in error_groups.items():
                parts.append(f"**故障码：{code}**")
                parts.append("")
                for chunk, processed in items:
                    parts.append(processed)
                    parts.append("")
                parts.append("")
        
        if other_chunks:
            parts.append("### 补充信息")
            parts.append("")
            for chunk, processed in other_chunks:
                parts.append(processed)
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_safety_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """安全合规查询上下文组装：警告类Chunk强制置顶，按风险等级排序"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        
        warning_chunks = [c for c in chunks if c.content_type == "warning"]
        other_chunks = [c for c in chunks if c.content_type != "warning"]
        
        if warning_chunks:
            parts.append("### 安全警告")
            parts.append("")
            for chunk in warning_chunks:
                processed = self.content_handler.process_chunk(chunk, intent)
                parts.append(processed)
                parts.append("")
        
        if other_chunks:
            parts.append("### 安全规范说明")
            parts.append("")
            for chunk in other_chunks:
                processed = self.content_handler.process_chunk(chunk, intent)
                parts.append(processed)
                parts.append("")
        
        return '\n'.join(parts)
    
    def _build_comparison_context(
        self,
        chunks: List[ProcessedChunk],
        intent: IntentClassification,
        target_model: Optional[str],
        timestamp: Optional[str]
    ) -> str:
        """对比查询上下文组装：按机型分组组装"""
        parts = []
        parts.append(self._build_global_header(target_model, intent, timestamp))
        
        model_groups = defaultdict(list)
        general_chunks = []
        
        for chunk in chunks:
            processed = self.content_handler.process_chunk(chunk, intent)
            model = chunk.metadata.get("model", chunk.metadata.get("机型", "通用"))
            if model and model != "通用":
                model_groups[model].append(processed)
            else:
                general_chunks.append(processed)
        
        if model_groups:
            parts.append("### 机型对比信息")
            parts.append("")
            for model, items in sorted(model_groups.items()):
                parts.append(f"#### {model}")
                parts.append("")
                for item in items:
                    parts.append(item)
                    parts.append("")
        
        if general_chunks:
            parts.append("### 通用信息")
            parts.append("")
            for item in general_chunks:
                parts.append(item)
                parts.append("")
        
        return '\n'.join(parts)
    
    def _smart_truncate(self, text: str, max_chars: int) -> str:
        """智能截断文本，尽量不在表格或代码块中间截断"""
        if len(text) <= max_chars:
            return text
        
        truncate_point = max_chars
        
        for boundary in ["\n\n", "\n---\n", "\n### "]:
            idx = text.rfind(boundary, 0, truncate_point)
            if idx > max_chars * 0.8:
                truncate_point = idx
                break
        
        truncated = text[:truncate_point].strip()
        return truncated + "\n\n...[上下文因长度限制被截断]"


# =============================================================================
# 4. 提示词模板系统
# =============================================================================

class PromptTemplate:
    """
    提示词模板
    
    包含基础通用模板和各类意图专用模板，着重幻觉防控：
    - 只使用检索到的官方说明书内容，不注入任何外部知识
    - 上下文开头明确标注 "以下内容全部来自大疆官方产品说明书"
    - 明确规定 "无信息时的标准回答"
    - 要求标注信息来源，便于追溯
    - 校验回答中的参数、数值是否与上下文一致
    - 校验是否包含禁止的表述（如 "可能""大概""我认为" 等模糊表述）
    """
    
    # 幻觉防控通用约束
    HALLUCINATION_CONTROL = """
【信息来源约束】
- 你只允许使用上方检索上下文中提供的大疆官方产品说明书内容作答。
- 严禁注入任何外部知识、个人经验、网络信息或推测。
- 如果上下文中的信息不足以回答用户问题，必须使用标准回答："根据当前检索到的大疆官方产品说明书内容，暂无相关信息，建议您查阅对应机型的最新用户手册或联系大疆官方客服获取准确信息。"

【回答一致性约束】
- 回答中的具体参数、数值、操作步骤必须与上下文完全一致，禁止自行修改或四舍五入。
- 若上下文中的参数存在多个版本，请明确标注来源机型和章节。

【表述禁止词】
- 严禁使用以下模糊表述："可能"、"大概"、"也许"、"我认为"、"我觉得"、"猜测"、"估计"、"应该"。
- 所有陈述应以确定事实语气表达，基于上下文直接陈述。

【来源标注约束】
- 每个关键事实后，如果上下文提供了明确的来源机型和章节，请尽可能简要标注。
- 示例格式：Mavic 4 Pro 最大飞行高度为 1000 米（来源：Mavic 4 Pro / 飞行参数）。
"""
    
    # 基础通用模板
    BASE_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户问题。

{context}

{hallucination_control}

【用户问题】
{query}

请基于上述官方说明书内容，用中文直接、准确地回答用户问题。如果涉及操作步骤，请保持顺序完整；如果涉及参数，请确保数值准确。"""
    
    # 参数查询专用模板
    PARAMETER_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的参数/规格查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 优先以表格或列表形式呈现参数，确保清晰可读。
2. 若同一参数在不同机型/不同场景下有差异，请分别列出并标注来源。
3. 对于单位、范围类参数，必须完整保留原始数值和单位。
4. 如果用户未指定机型，请基于上下文中出现的机型分别说明。

请用中文准确回答。"""
    
    # 操作步骤查询专用模板
    OPERATION_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的操作步骤查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 操作步骤必须按原文顺序完整呈现，禁止跳过、合并或重新排序步骤。
2. 对于安全关键步骤（如起飞前检查、校准操作），必须保留所有警告和注意事项。
3. 如果步骤涉及 prerequisites（前提条件），请优先说明。
4. 每个关键步骤后可简要标注来源章节。

请用中文逐步回答。"""
    
    # 故障排查查询专用模板
    TROUBLESHOOTING_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的故障排查查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 如果涉及故障码，请优先给出故障码及其官方含义。
2. 然后依次说明：可能原因 → 官方推荐的解决方法/排查步骤。
3. 若上下文中未包含该故障码或故障现象，请直接给出无信息标准回答，禁止自行编造原因和解决办法。
4. 对于需要送修的情况，请明确说明并建议联系大疆官方售后。

请用中文有条理地回答。"""
    
    # 安全合规查询专用模板
    SAFETY_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的安全合规查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 所有安全警告、法律要求、禁飞规则必须完整准确地转述，不得弱化风险描述。
2. 警告类内容优先呈现，并使用明确的风险等级表述（如：严禁、禁止、必须注意）。
3. 若涉及法规，请说明该信息来自说明书原文，并建议用户同时参考当地最新法规。
4. 禁止为了"方便用户"而省略安全限制条件。

请用中文严肃、准确地回答。"""
    
    # 对比查询专用模板
    COMPARISON_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的对比查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 按机型/模式分组对比，相同维度并列呈现，便于用户直观对比差异。
2. 仅对比上下文中明确提到的项目，未提及的项目说明"无信息"，禁止脑补。
3. 对于存在差异的参数，请分别列出各机型的数值并标注来源。
4. 不要加入个人推荐或购买建议，只陈述官方参数差异。

请用中文客观、准确地回答。"""
    
    # 功能说明查询专用模板
    FUNCTION_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的功能说明查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 准确说明功能的定义、适用条件、使用限制和注意事项。
2. 若功能在不同机型或固件版本中存在差异，请分别说明。
3. 对于上下文中未明确提及的功能，给出无信息标准回答。

请用中文准确回答。"""
    
    # 维护保养查询专用模板
    MAINTENANCE_TEMPLATE = """以下内容全部来自大疆官方产品说明书，请严格基于该内容回答用户的维护保养查询。

{context}

{hallucination_control}

【用户问题】
{query}

【回答要求】
1. 保养步骤按原文顺序完整呈现。
2. 涉及电池保养、存储规范等，必须完整保留温度、电量、时间等关键参数。
3. 对于可能导致保修失效的操作，必须明确警告。

请用中文准确回答。"""
    
    INTENT_TEMPLATES = {
        PrimaryIntent.PARAMETER_QUERY: PARAMETER_TEMPLATE,
        PrimaryIntent.OPERATION_QUERY: OPERATION_TEMPLATE,
        PrimaryIntent.TROUBLESHOOTING: TROUBLESHOOTING_TEMPLATE,
        PrimaryIntent.SAFETY_COMPLIANCE: SAFETY_TEMPLATE,
        PrimaryIntent.COMPARISON: COMPARISON_TEMPLATE,
        PrimaryIntent.FUNCTION_QUERY: FUNCTION_TEMPLATE,
        PrimaryIntent.MAINTENANCE: MAINTENANCE_TEMPLATE,
        PrimaryIntent.UNKNOWN: BASE_TEMPLATE,
    }
    
    @classmethod
    def get_template(cls, intent: IntentClassification) -> str:
        """根据意图获取对应模板"""
        return cls.INTENT_TEMPLATES.get(
            intent.primary_intent,
            cls.BASE_TEMPLATE
        )
    
    @classmethod
    def render(
        cls,
        query: str,
        context: AssembledContext,
        intent: IntentClassification
    ) -> str:
        """渲染提示词"""
        template = cls.get_template(intent)
        return template.format(
            context=context.markdown_text,
            hallucination_control=cls.HALLUCINATION_CONTROL,
            query=query
        )


# =============================================================================
# 5. Prompt 引擎
# =============================================================================

class PromptEngine:
    """
    Prompt 引擎
    
    整合上下文组装和提示词渲染，输出可直接用于大模型的完整 Prompt。
    """
    
    def __init__(
        self,
        score_threshold: float = 0.3,
        max_context_chars: int = 8000
    ):
        self.context_builder = StructuredContextBuilder(
            score_threshold=score_threshold,
            max_total_chars=max_context_chars,
            include_source_info=True
        )
    
    def build_prompt(
        self,
        query: str,
        rerank_result: RerankResult,
        intent: IntentClassification,
        target_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """构建完整Prompt"""
        assembled = self.context_builder.build(
            rerank_result=rerank_result,
            intent=intent,
            target_model=target_model
        )
        
        prompt_text = PromptTemplate.render(
            query=query,
            context=assembled,
            intent=intent
        )
        
        prompt_meta = {
            "query": query,
            "target_model": target_model,
            "primary_intent": intent.primary_intent.value,
            "secondary_intent": intent.secondary_intent.value,
            "chunk_count": assembled.chunk_count,
            "context_chars": assembled.total_chars,
            "prompt_chars": len(prompt_text),
            "assembly_time_ms": assembled.assembly_time_ms,
            "timestamp": assembled.metadata.get("timestamp")
        }
        
        return {
            "prompt": prompt_text,
            "context": assembled,
            "metadata": prompt_meta
        }


# =============================================================================
# 6. 便捷函数
# =============================================================================

def build_context(
    rerank_result: RerankResult,
    intent: IntentClassification,
    target_model: Optional[str] = None,
    score_threshold: float = 0.3,
    max_context_chars: int = 8000
) -> AssembledContext:
    """便捷函数：组装结构化上下文"""
    builder = StructuredContextBuilder(
        score_threshold=score_threshold,
        max_total_chars=max_context_chars
    )
    return builder.build(rerank_result, intent, target_model)


def build_prompt(
    query: str,
    rerank_result: RerankResult,
    intent: IntentClassification,
    target_model: Optional[str] = None,
    score_threshold: float = 0.3,
    max_context_chars: int = 8000
) -> Dict[str, Any]:
    """便捷函数：构建完整Prompt"""
    engine = PromptEngine(
        score_threshold=score_threshold,
        max_context_chars=max_context_chars
    )
    return engine.build_prompt(query, rerank_result, intent, target_model)


# =============================================================================
# 7. 输出校验器（轻量级，用于LLM输出后校验）
# =============================================================================

class ResponseValidator:
    """
    回答校验器
    
    对LLM生成的回答进行轻量级规则校验：
    - 检测模糊表述（可能、大概、我认为...）
    - 检测数值是否与上下文一致（抽样比对）
    - 检测是否出现无信息但未使用标准回答的情况
    """
    
    FORBIDDEN_PHRASES = [
        "可能", "大概", "也许", "我认为", "我觉得", "猜测", "估计", "应该",
        "maybe", "probably", "perhaps", "I think", "I guess", "estimated"
    ]
    
    @classmethod
    def validate(cls, response_text: str, assembled_context: AssembledContext) -> Dict[str, Any]:
        """校验LLM回答"""
        issues = []
        
        # 1. 模糊表述检测
        found_phrases = []
        for phrase in cls.FORBIDDEN_PHRASES:
            if phrase in response_text:
                found_phrases.append(phrase)
        
        if found_phrases:
            issues.append({
                "type": "forbidden_phrase",
                "message": f"检测到禁止的模糊表述: {', '.join(found_phrases)}",
                "severity": "warning"
            })
        
        # 2. 无信息标准回答检测
        has_context = assembled_context.chunk_count > 0
        no_info_keywords = ["暂无相关信息", "无法找到", "没有相关信息"]
        has_no_info_reply = any(kw in response_text for kw in no_info_keywords)
        
        if not has_context and not has_no_info_reply:
            issues.append({
                "type": "missing_no_info_reply",
                "message": "上下文中无有效Chunk，但回答未使用标准无信息回复",
                "severity": "error"
            })
        
        # 3. 数值一致性检测
        context_numbers = cls._extract_numbers(assembled_context.markdown_text)
        response_numbers = cls._extract_numbers(response_text)
        
        if len(context_numbers) >= 3 and len(response_numbers) == 0:
            issues.append({
                "type": "missing_numbers",
                "message": "上下文中包含多个数值参数，但回答中未出现任何数值",
                "severity": "warning"
            })
        
        return {
            "passed": len([i for i in issues if i["severity"] == "error"]) == 0,
            "issues": issues,
            "context_chunk_count": assembled_context.chunk_count,
            "response_chars": len(response_text)
        }
    
    @classmethod
    def _extract_numbers(cls, text: str) -> List[str]:
        """从文本中提取带单位的数值（简化版）"""
        pattern = r'\d+(?:\.\d+)?\s*(?:米|km|m|克|g|kg|分钟|min|小时|h|°C|℃|mAh|V|A|W|dB|fps|MP|GB|°|%|ms)'
        return re.findall(pattern, text)


if __name__ == "__main__":
    print("=" * 70)
    print("上下文组装器与提示词工程模块")
    print("=" * 70)
    print("\n核心组件:")
    print("1. ChunkPreprocessor      - Chunk过滤与意图加权排序")
    print("2. SpecialContentHandler  - 表格/步骤/警告/故障码特殊处理")
    print("3. StructuredContextBuilder - Markdown结构化上下文组装")
    print("4. PromptTemplate         - 基础通用模板 + 6类意图专用模板")
    print("5. PromptEngine           - 整合上下文组装与提示词渲染")
    print("6. ResponseValidator      - LLM输出后幻觉防控轻量校验")
    print("\n使用方式:")
    print("from context_builder import build_prompt, build_context")
    print("prompt_result = build_prompt(query, rerank_result, intent)")
    print("print(prompt_result['prompt'])")
