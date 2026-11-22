"""
生成流水线 (Generation Pipeline)

功能：
1. DJIRAGPipeline 流水线管理器，协调各组件执行顺序
2. 处理异常情况，记录全链路日志
3. 基于评估结果筛选上下文
4. 答案草稿生成与审查
5. 生成后处理（引用标记、格式美化、免责声明）
"""

import re
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# 导入已有模块
from rag_pipeline import RAGRetrievalPipeline, PipelineResult, GenerationContext
from retrieval_system import IntelligentRetrievalSystem, RetrievalContext
from multi_path_retriever import MultiPathRetriever, MultiPathResult
from reranker import create_reranker, RerankResult, RerankedChunk
from context_builder import PromptEngine, AssembledContext, ResponseValidator
from llm_generator import (
    BaseLLMGenerator, create_generator,
    GenerationConfig, GenerationResult
)
from config_center import ConfigCenter, setup_logging
from intent_classifier import PrimaryIntent, IntentClassification


logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class ContextFilterResult:
    """上下文筛选结果"""
    filtered_chunks: List[Dict[str, Any]]
    removed_chunks: List[Dict[str, Any]]
    filter_reasons: List[str] = field(default_factory=list)
    doc_id_set: set = field(default_factory=set)


@dataclass
class DraftAnswer:
    """答案草稿"""
    text: str
    raw_generation: GenerationResult
    passed_validation: bool
    validation_issues: List[Dict[str, Any]] = field(default_factory=list)
    citation_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class FinalAnswer:
    """最终答案"""
    text: str
    answer_type: str  # "draft" | "reviewed"
    citations: List[str] = field(default_factory=list)
    disclaimer: str = ""
    post_processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DJIPipelineResult:
    """
    DJI RAG 流水线完整结果
    
    包含检索阶段和生成阶段的全部信息
    """
    # 检索结果（复用已有结构）
    retrieval_result: PipelineResult
    
    # 生成阶段结果
    generation_result: Optional[GenerationResult] = None
    draft_answer: Optional[DraftAnswer] = None
    final_answer: Optional[FinalAnswer] = None
    
    # 筛选结果
    context_filter: Optional[ContextFilterResult] = None
    
    # 全链路日志
    execution_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间统计
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # 时间戳
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# =============================================================================
# 日志记录器
# =============================================================================

class PipelineLogger:
    """流水线日志记录器"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.logs: List[Dict[str, Any]] = []
    
    def log(self, stage: str, event: str, details: Optional[Dict] = None, level: str = "info"):
        """记录日志"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "event": event,
            "level": level,
            "details": details or {}
        }
        self.logs.append(entry)
        
        if not self.enabled:
            return
        
        msg = f"[{stage}] {event}"
        if details:
            msg += f" | {json.dumps(details, ensure_ascii=False, default=str)}"
        
        if level == "error":
            logger.error(msg)
        elif level == "warning":
            logger.warning(msg)
        elif level == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """获取所有日志"""
        return self.logs.copy()
    
    def get_stage_logs(self, stage: str) -> List[Dict[str, Any]]:
        """获取指定阶段的日志"""
        return [l for l in self.logs if l["stage"] == stage]


# =============================================================================
# 上下文筛选器
# =============================================================================

class ContextFilter:
    """
    上下文筛选器
    
    基于评估结果筛选上下文：
    - 基于文档ID去重与筛选
    - 检测文档标题行等噪声内容
    """
    
    def __init__(self):
        self.title_patterns = [
            r'^第[一二三四五六七八九十\d]+章\s*',
            r'^\d+\.\d+\s+',
            r'^(?:目录|索引|附录|前言|简介)\s*$',
            r'^\s*DJI\s*\w+\s*User\s*Manual\s*$',
        ]
    
    def filter(
        self,
        rerank_result: RerankResult,
        min_score: float = 0.3,
        max_chunks: int = 10
    ) -> ContextFilterResult:
        """
        筛选上下文
        
        Args:
            rerank_result: 重排序结果
            min_score: 最低分数阈值
            max_chunks: 最大保留chunk数
            
        Returns:
            ContextFilterResult
        """
        filtered = []
        removed = []
        reasons = []
        doc_id_set = set()
        
        for chunk in rerank_result.reranked_chunks:
            chunk_id = chunk.chunk_id
            doc_text = chunk.document or ""
            score = chunk.final_score
            
            # 1. 分数过滤
            if score < min_score:
                removed.append({"chunk_id": chunk_id, "reason": "低分过滤", "score": score})
                continue
            
            # 2. 文档ID去重（同一文档保留最高分）
            doc_key = self._extract_doc_key(chunk)
            if doc_key in doc_id_set:
                removed.append({"chunk_id": chunk_id, "reason": "文档去重", "doc_key": doc_key})
                continue
            
            # 3. 标题行/噪声检测
            if self._is_title_only(doc_text):
                removed.append({"chunk_id": chunk_id, "reason": "标题/噪声内容"})
                continue
            
            # 4. 内容重复度检测（简单版本）
            if self._is_redundant(doc_text, filtered):
                removed.append({"chunk_id": chunk_id, "reason": "内容冗余"})
                continue
            
            doc_id_set.add(doc_key)
            filtered.append({
                "chunk_id": chunk_id,
                "document": doc_text,
                "metadata": chunk.metadata,
                "score": score,
                "doc_key": doc_key
            })
        
        # 限制数量
        if len(filtered) > max_chunks:
            removed.extend([
                {"chunk_id": c["chunk_id"], "reason": "超出数量限制"}
                for c in filtered[max_chunks:]
            ])
            filtered = filtered[:max_chunks]
        
        reasons = list(set(r["reason"] for r in removed))
        
        return ContextFilterResult(
            filtered_chunks=filtered,
            removed_chunks=removed,
            filter_reasons=reasons,
            doc_id_set=doc_id_set
        )
    
    def _extract_doc_key(self, chunk: RerankedChunk) -> str:
        """提取文档去重键"""
        metadata = chunk.metadata or {}
        model = metadata.get("model", metadata.get("机型", ""))
        section = metadata.get("section", "")
        # 使用机型+章节作为去重键
        return f"{model}:{section}"
    
    def _is_title_only(self, text: str) -> bool:
        """检测是否仅为标题"""
        text = text.strip()
        if len(text) < 10:
            return True
        for pattern in self.title_patterns:
            if re.match(pattern, text):
                # 如果匹配标题模式且内容很短，认为是标题
                if len(text) < 50:
                    return True
        return False
    
    def _is_redundant(self, text: str, existing: List[Dict]) -> bool:
        """检测内容是否冗余（与已有内容高度相似）"""
        text_lower = text.lower()
        for item in existing:
            existing_text = item["document"].lower()
            # 简单判断：如果新内容完全包含于已有内容中
            if text_lower in existing_text and len(text_lower) < len(existing_text) * 0.8:
                return True
            # 或已有内容完全包含于新内容中（保留更长的）
            if existing_text in text_lower and len(existing_text) < len(text_lower) * 0.8:
                # 替换为更长的
                item["document"] = text
                return True
        return False


# =============================================================================
# 答案后处理器
# =============================================================================

class AnswerPostProcessor:
    """
    答案后处理器
    
    功能：
    1. 添加引用标记
    2. 添加格式美化
    3. 添加免责声明
    """
    
    DISCLAIMER_TEXT = (
        "\n\n---\n"
        "**免责声明**：以上回答基于大疆官方产品说明书检索结果生成，"
        "仅供参考。具体操作请以对应机型的最新用户手册为准，"
        "或联系大疆官方客服获取准确信息。"
    )
    
    def __init__(
        self,
        add_citations: bool = True,
        add_disclaimer: bool = True,
        citation_format: str = "[{index}]"
    ):
        self.add_citations = add_citations
        self.add_disclaimer = add_disclaimer
        self.citation_format = citation_format
    
    def process(
        self,
        answer_text: str,
        context_chunks: List[Dict[str, Any]],
        intent: Optional[IntentClassification] = None
    ) -> FinalAnswer:
        """
        后处理答案
        
        Args:
            answer_text: 原始答案文本
            context_chunks: 使用的上下文chunks
            intent: 意图分类
            
        Returns:
            FinalAnswer
        """
        citations = []
        processed_text = answer_text
        
        # 1. 添加引用标记
        if self.add_citations and context_chunks:
            processed_text, citations = self._add_citations(
                processed_text, context_chunks
            )
        
        # 2. 格式美化
        processed_text = self._beautify_format(processed_text, intent)
        
        # 3. 添加免责声明
        disclaimer = ""
        if self.add_disclaimer:
            disclaimer = self.DISCLAIMER_TEXT
            processed_text += disclaimer
        
        return FinalAnswer(
            text=processed_text,
            answer_type="draft",
            citations=citations,
            disclaimer=disclaimer,
            post_processed=True,
            metadata={
                "citation_count": len(citations),
                "has_disclaimer": self.add_disclaimer
            }
        )
    
    def _add_citations(
        self,
        text: str,
        chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        """
        添加引用标记
        
        策略：在答案中检测与chunk内容相关的句子，添加引用标记
        """
        citations = []
        citation_map = {}  # chunk_id -> index
        
        for i, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get("document", "")
            metadata = chunk.get("metadata", {})
            model = metadata.get("model", metadata.get("机型", "未知机型"))
            section = metadata.get("section", "未知章节")
            
            # 提取chunk中的关键短语（长度5-20的子串）用于匹配
            key_phrases = self._extract_key_phrases(chunk_text)
            
            cited = False
            for phrase in key_phrases:
                if phrase in text and len(phrase) > 5:
                    # 在短语后添加引用标记
                    # 避免重复添加
                    marker = self.citation_format.format(index=i)
                    # 找到短语在文本中的位置，在其所在句子末尾添加标记
                    text = self._insert_citation(text, phrase, marker)
                    cited = True
                    break  # 每个chunk只添加一次引用
            
            if cited:
                citation_text = f"{model} / {section}"
                citations.append(citation_text)
                citation_map[chunk.get("chunk_id", "")] = citation_text
        
        return text, citations
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """提取关键短语"""
        # 按句子分割，过滤短句，返回较长的短语
        sentences = re.split(r'[。！？\n]', text)
        phrases = []
        for s in sentences:
            s = s.strip()
            if len(s) >= 10:
                phrases.append(s)
            if len(phrases) >= max_phrases:
                break
        return phrases
    
    def _insert_citation(self, text: str, phrase: str, marker: str) -> str:
        """在短语所在位置插入引用标记"""
        # 简单策略：在短语后面直接插入标记
        idx = text.find(phrase)
        if idx == -1:
            return text
        
        end_idx = idx + len(phrase)
        # 确保不重复插入
        after_text = text[end_idx:]
        if marker in text[max(0, end_idx-10):end_idx+10]:
            return text
        
        return text[:end_idx] + marker + text[end_idx:]
    
    def _beautify_format(self, text: str, intent: Optional[IntentClassification]) -> str:
        """格式美化"""
        # 确保段落之间有换行
        text = re.sub(r'([^\n])\n([^\n])', r'\1\n\2', text)
        
        # 参数查询：尝试将键值对格式化为列表
        if intent and intent.primary_intent == PrimaryIntent.PARAMETER_QUERY:
            text = self._beautify_parameters(text)
        
        # 操作步骤查询：确保步骤编号格式统一
        if intent and intent.primary_intent == PrimaryIntent.OPERATION_QUERY:
            text = self._beautify_steps(text)
        
        # 故障排查：统一故障码格式
        if intent and intent.primary_intent == PrimaryIntent.TROUBLESHOOTING:
            text = self._beautify_troubleshooting(text)
        
        return text.strip()
    
    def _beautify_parameters(self, text: str) -> str:
        """美化参数格式"""
        # 将 "参数: 值" 格式化为加粗
        text = re.sub(r'([\u4e00-\u9fa5\w\s]+)[：:]\s*(.+?)(?=\n|$)', r'**\1**：\2', text)
        return text
    
    def _beautify_steps(self, text: str) -> str:
        """美化步骤格式"""
        # 统一步骤编号为 1. 2. 3.
        lines = text.split('\n')
        new_lines = []
        for line in lines:
            # 匹配各种编号格式
            match = re.match(r'^(?:步骤\s*)?[（(]?(\d+)[)）]?[\.、\s]*(.*)$', line.strip())
            if match:
                new_lines.append(f"{match.group(1)}. {match.group(2)}")
            else:
                new_lines.append(line)
        return '\n'.join(new_lines)
    
    def _beautify_troubleshooting(self, text: str) -> str:
        """美化故障排查格式"""
        # 加粗故障码
        text = re.sub(r'(故障码[：:]\s*)([A-Z]?\d+)', r'\1**\2**', text)
        text = re.sub(r'(错误码[：:]\s*)([A-Z]?\d+)', r'\1**\2**', text)
        return text


# =============================================================================
# DJI RAG 流水线
# =============================================================================

class DJIRAGPipeline:
    """
    DJI RAG 完整流水线
    
    协调以下组件的执行顺序：
    1. 查询理解与增强
    2. 多路检索重排
    3. 上下文组装
    4. 提示词模板
    5. 大模型生成
    6. 答案后处理
    
    处理异常，记录全链路日志
    """

    def __init__(
        self,
        config: Optional[ConfigCenter] = None,
        retrieval_pipeline: Optional[RAGRetrievalPipeline] = None,
        llm_generator: Optional[BaseLLMGenerator] = None
    ):
        """
        初始化流水线
        
        Args:
            config: 配置中心，None则自动加载
            retrieval_pipeline: 检索流水线，None则根据配置创建
            llm_generator: LLM生成器，None则根据配置创建
        """
        # 配置
        self.config = config or ConfigCenter()
        
        # 日志
        setup_logging(self.config)
        self.pipeline_logger = PipelineLogger(
            enabled=self.config.pipeline.enable_logging
        )
        
        self.pipeline_logger.log("init", "流水线初始化开始", {
            "config_path": getattr(self.config, '_config_path', 'default'),
            "llm_backend": self.config.llm.backend
        })
        
        # 检索流水线
        if retrieval_pipeline:
            self.retrieval_pipeline = retrieval_pipeline
        else:
            self.retrieval_pipeline = self._create_retrieval_pipeline()
        
        # LLM生成器
        if llm_generator:
            self.llm_generator = llm_generator
        else:
            self.llm_generator = self._create_llm_generator()
        
        # 辅助组件
        self.context_filter = ContextFilter()
        self.post_processor = AnswerPostProcessor(
            add_citations=self.config.pipeline.add_citations,
            add_disclaimer=self.config.pipeline.add_disclaimer
        )
        
        self.pipeline_logger.log("init", "流水线初始化完成")
    
    def _create_retrieval_pipeline(self) -> RAGRetrievalPipeline:
        """创建检索流水线"""
        from vector_db_system import (
            ChromaDBManager, EmbeddingModelManager,
            ChromaConfig, EmbeddingConfig
        )
        
        cfg = self.config.retrieval
        
        self.pipeline_logger.log("init", "创建检索流水线", {
            "chroma_db_path": cfg.chroma_db_path,
            "reranker_type": cfg.reranker_type
        })
        
        # 初始化嵌入模型
        embed_config = EmbeddingConfig(
            model_type=cfg.embedding_model_type,
            precomputed_cache_path=cfg.embedding_cache_path
        )
        embedding_model = EmbeddingModelManager(embed_config)
        
        # 初始化ChromaDB
        chroma_config = ChromaConfig(
            persist_path=cfg.chroma_db_path,
            collection_name=cfg.collection_name
        )
        chroma_db = ChromaDBManager(chroma_config)
        chroma_db.create_or_get_collection()
        
        # 创建检索流水线
        return RAGRetrievalPipeline(
            chroma_db=chroma_db,
            embedding_model=embedding_model,
            reranker_type=cfg.reranker_type,
            reranker_threshold=cfg.reranker_threshold,
            multipath_top_k=cfg.multipath_top_k,
            final_top_k=cfg.final_top_k
        )
    
    def _create_llm_generator(self) -> BaseLLMGenerator:
        """创建LLM生成器"""
        cfg = self.config.llm
        
        self.pipeline_logger.log("init", "创建LLM生成器", {
            "backend": cfg.backend,
            "model": cfg.model_name
        })
        
        gen = create_generator(
            backend=cfg.backend,
            model_name=cfg.model_name,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            vllm_mode=getattr(cfg, 'vllm_mode', 'server'),
            device=getattr(cfg, 'device', 'cuda'),
            system_prompt=cfg.system_prompt
        )
        
        if not gen._initialized:
            self.pipeline_logger.log(
                "init", "LLM生成器初始化失败，后续生成将不可用",
                level="warning"
            )
        
        return gen
    
    def run(self, query: str, require_json: bool = False) -> DJIPipelineResult:
        """
        执行完整RAG流水线（检索 + 生成）
        
        Args:
            query: 用户查询
            require_json: 是否要求JSON格式输出
            
        Returns:
            DJIPipelineResult
        """
        total_start = time.time()
        exec_log = []
        
        def log(stage: str, event: str, details: Optional[Dict] = None, level: str = "info"):
            self.pipeline_logger.log(stage, event, details, level)
            exec_log.append({
                "timestamp": time.time(),
                "stage": stage,
                "event": event,
                "details": details,
                "level": level
            })
        
        log("start", "流水线开始", {"query": query})
        
        # ========== Stage 1: 检索阶段 ==========
        retrieval_result = None
        try:
            log("retrieval", "开始检索")
            stage_start = time.time()
            retrieval_result = self.retrieval_pipeline.search(query)
            stage_time = (time.time() - stage_start) * 1000
            log("retrieval", "检索完成", {
                "time_ms": stage_time,
                "recall_chunks": retrieval_result.multipath_result.total_unique_chunks,
                "final_docs": len(retrieval_result.final_documents)
            })
        except Exception as e:
            log("retrieval", "检索失败", {"error": str(e)}, level="error")
            # 构造空结果
            retrieval_result = self._empty_retrieval_result(query)
        
        # ========== Stage 2: 上下文筛选 ==========
        filter_result = None
        try:
            log("filter", "开始上下文筛选")
            if retrieval_result and retrieval_result.rerank_result:
                filter_result = self.context_filter.filter(
                    retrieval_result.rerank_result,
                    min_score=self.config.pipeline.score_threshold,
                    max_chunks=self.config.retrieval.final_top_k
                )
                log("filter", "筛选完成", {
                    "filtered": len(filter_result.filtered_chunks),
                    "removed": len(filter_result.removed_chunks),
                    "reasons": filter_result.filter_reasons
                })
        except Exception as e:
            log("filter", "筛选失败", {"error": str(e)}, level="warning")
        
        # ========== Stage 3: 生成阶段 ==========
        generation_result = None
        draft_answer = None
        final_answer = None
        
        if retrieval_result and retrieval_result.generation_context:
            try:
                log("generation", "开始生成")
                stage_start = time.time()
                
                # 构建 prompt
                prompt = retrieval_result.generation_context.prompt
                
                # 如果有筛选结果，基于筛选后的 chunk 重新组装 prompt
                if filter_result and filter_result.filtered_chunks:
                    prompt = self._rebuild_prompt_with_filter(
                        query,
                        retrieval_result,
                        filter_result
                    )
                
                # 生成配置
                gen_cfg = GenerationConfig(
                    temperature=self.config.llm.temperature,
                    top_p=self.config.llm.top_p,
                    max_tokens=self.config.llm.max_tokens,
                    frequency_penalty=self.config.llm.frequency_penalty,
                    response_format="json" if require_json else None
                )
                
                # 调用 LLM 生成（带重试）
                generation_result = self._generate_with_retry(prompt, gen_cfg)
                
                gen_time = (time.time() - stage_start) * 1000
                log("generation", "生成完成", {
                    "time_ms": gen_time,
                    "prompt_tokens": generation_result.prompt_tokens,
                    "completion_tokens": generation_result.completion_tokens
                })
                
                # ========== Stage 4: 审查草稿 ==========
                log("review", "开始审查草稿")
                draft_answer = self._create_draft_answer(
                    generation_result,
                    retrieval_result.generation_context.assembled_context
                )
                log("review", "审查完成", {
                    "passed": draft_answer.passed_validation,
                    "issues": len(draft_answer.validation_issues)
                })
                
                # ========== Stage 5: 生成最终答案 ==========
                log("finalize", "生成最终答案")
                context_chunks = (
                    filter_result.filtered_chunks if filter_result
                    else retrieval_result.final_documents
                )
                final_answer = self.post_processor.process(
                    draft_answer.text,
                    context_chunks,
                    retrieval_result.context.intent
                )
                
                # 如果审查通过，标记为已审查
                if draft_answer.passed_validation:
                    final_answer.answer_type = "reviewed"
                
                log("finalize", "最终答案生成完成", {
                    "type": final_answer.answer_type,
                    "citations": len(final_answer.citations),
                    "has_disclaimer": bool(final_answer.disclaimer)
                })
                
            except Exception as e:
                log("generation", "生成阶段失败", {"error": str(e)}, level="error")
                final_answer = FinalAnswer(
                    text=f"抱歉，生成回答时发生错误：{str(e)}",
                    answer_type="error",
                    post_processed=False
                )
        else:
            log("generation", "无检索结果，跳过生成", level="warning")
            final_answer = FinalAnswer(
                text="根据当前检索到的大疆官方产品说明书内容，暂无相关信息，"
                     "建议您查阅对应机型的最新用户手册或联系大疆官方客服获取准确信息。",
                answer_type="no_info",
                post_processed=True
            )
        
        total_time = (time.time() - total_start) * 1000
        gen_time = generation_result.generation_time_ms if generation_result else 0.0
        
        log("complete", "流水线完成", {"total_time_ms": total_time})
        
        return DJIPipelineResult(
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            draft_answer=draft_answer,
            final_answer=final_answer,
            context_filter=filter_result,
            execution_log=exec_log,
            generation_time_ms=gen_time,
            total_time_ms=total_time
        )
    
    def _rebuild_prompt_with_filter(
        self,
        query: str,
        retrieval_result: PipelineResult,
        filter_result: ContextFilterResult
    ) -> str:
        """基于筛选结果重新构建 prompt"""
        # 简单策略：替换上下文部分
        original_prompt = retrieval_result.generation_context.prompt
        
        # 构建新的上下文文本
        context_parts = []
        for i, chunk in enumerate(filter_result.filtered_chunks, start=1):
            context_parts.append(f"### 分块内容 {i}")
            context_parts.append("")
            metadata = chunk.get("metadata", {})
            model = metadata.get("model", metadata.get("机型", "未知"))
            section = metadata.get("section", "未知章节")
            context_parts.append(f"- **来源机型**：{model}")
            context_parts.append(f"- **来源章节**：{section}")
            context_parts.append("")
            context_parts.append("**内容**：")
            context_parts.append(chunk.get("document", ""))
            context_parts.append("")
            context_parts.append("---")
            context_parts.append("")
        
        new_context = "\n".join(context_parts)
        
        # 替换原始 prompt 中的上下文部分
        # 策略：找到 "## 检索上下文信息" 到 "【用户问题】" 之间的内容替换
        pattern = r'(## 检索上下文信息.*?)\n\n【用户问题】'
        if re.search(pattern, original_prompt, re.DOTALL):
            new_prompt = re.sub(
                pattern,
                f"## 检索上下文信息\n\n{new_context}\n\n【用户问题】",
                original_prompt,
                flags=re.DOTALL
            )
            return new_prompt
        
        # 如果无法精确替换，返回原始 prompt
        return original_prompt
    
    def _generate_with_retry(
        self,
        prompt: str,
        config: GenerationConfig
    ) -> GenerationResult:
        """带重试的生成"""
        max_retries = self.config.pipeline.max_retries
        retry_delay = self.config.pipeline.retry_delay
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return self.llm_generator.generate(prompt, config)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"生成失败，{retry_delay}秒后重试 ({attempt+1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
                else:
                    raise last_error
        
        raise last_error or RuntimeError("生成失败")
    
    def _create_draft_answer(
        self,
        generation_result: GenerationResult,
        assembled_context: AssembledContext
    ) -> DraftAnswer:
        """创建答案草稿并进行校验"""
        text = generation_result.text
        
        # 使用 ResponseValidator 进行校验
        validation = ResponseValidator.validate(text, assembled_context)
        
        # 尝试提取 JSON（如果是 JSON 模式）
        citation_map = {}
        json_data = self.llm_generator.extract_json(text)
        if json_data and isinstance(json_data, dict):
            # 如果成功解析 JSON，使用 answer 字段
            if "answer" in json_data:
                text = json_data["answer"]
            if "sources" in json_data:
                citation_map["json_sources"] = json_data["sources"]
        
        return DraftAnswer(
            text=text,
            raw_generation=generation_result,
            passed_validation=validation.get("passed", False),
            validation_issues=validation.get("issues", []),
            citation_map=citation_map
        )
    
    def _empty_retrieval_result(self, query: str) -> PipelineResult:
        """构造空检索结果"""
        from multi_path_retriever import MultiPathResult
        from reranker import RerankResult
        
        return PipelineResult(
            query=query,
            context=RetrievalContext(
                original_query=query,
                enhanced=None,  # type: ignore
                intent=None,  # type: ignore
                filter_condition=None  # type: ignore
            ),
            multipath_result=MultiPathResult(
                query=query,
                enhanced_query=query,
                channel_results={},
                fused_results=[],
                total_channels=0,
                active_channels=0,
                total_unique_chunks=0,
                fusion_time_ms=0.0
            ),
            rerank_result=RerankResult(
                query=query,
                reranked_chunks=[],
                input_count=0,
                output_count=0,
                filtered_count=0,
                avg_score=0.0,
                max_score=0.0,
                min_score=0.0,
                rerank_time_ms=0.0,
                threshold=0.0
            ),
            final_documents=[],
            total_time_ms=0.0,
            stage_times={},
            generation_context=None
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        流水线健康检查
        
        Returns:
            各组件健康状态
        """
        llm_health = self.llm_generator.health_check()
        
        return {
            "llm": {
                "healthy": llm_health.healthy,
                "model": llm_health.model_name,
                "latency_ms": llm_health.latency_ms,
                "error": llm_health.error_message
            },
            "retrieval": {
                "healthy": True,  # 简化为始终健康，实际可检查ChromaDB连接
                "stats": self.retrieval_pipeline.get_stats()
            },
            "overall": llm_health.healthy  # 整体健康取决于LLM
        }
    
    def explain_result(self, result: DJIPipelineResult) -> str:
        """
        解释流水线结果
        
        Args:
            result: 流水线结果
            
        Returns:
            格式化说明字符串
        """
        lines = [
            "=" * 70,
            "DJI RAG 生成流水线结果",
            "=" * 70,
            f"",
            f"【原始查询】{result.retrieval_result.query}",
            f"",
            f"【检索阶段】",
            f"  召回Chunk数: {result.retrieval_result.multipath_result.total_unique_chunks}",
            f"  最终文档数: {len(result.retrieval_result.final_documents)}",
            f"  检索耗时: {result.retrieval_result.total_time_ms:.2f}ms",
        ]
        
        if result.context_filter:
            lines.extend([
                f"",
                f"【上下文筛选】",
                f"  保留: {len(result.context_filter.filtered_chunks)} chunks",
                f"  过滤: {len(result.context_filter.removed_chunks)} chunks",
                f"  过滤原因: {', '.join(result.context_filter.filter_reasons)}"
            ])
        
        if result.generation_result:
            lines.extend([
                f"",
                f"【生成阶段】",
                f"  生成耗时: {result.generation_result.generation_time_ms:.2f}ms",
                f"  Prompt Tokens: {result.generation_result.prompt_tokens}",
                f"  Completion Tokens: {result.generation_result.completion_tokens}",
                f"  Finish Reason: {result.generation_result.finish_reason}"
            ])
        
        if result.draft_answer:
            lines.extend([
                f"",
                f"【草稿审查】",
                f"  通过校验: {result.draft_answer.passed_validation}",
                f"  问题数: {len(result.draft_answer.validation_issues)}"
            ])
            for issue in result.draft_answer.validation_issues:
                lines.append(f"    - [{issue.get('severity', 'info')}] {issue.get('message', '')}")
        
        if result.final_answer:
            lines.extend([
                f"",
                f"【最终答案】",
                f"  类型: {result.final_answer.answer_type}",
                f"  引用数: {len(result.final_answer.citations)}",
                f"  后处理: {result.final_answer.post_processed}",
                f"  长度: {len(result.final_answer.text)} 字符"
            ])
        
        lines.extend([
            f"",
            f"【总耗时】{result.total_time_ms:.2f}ms",
            f"【时间戳】{result.timestamp}",
            "=" * 70
        ])
        
        return "\n".join(lines)


# =============================================================================
# 便捷函数
# =============================================================================

def create_generation_pipeline(
    config_path: Optional[str] = None,
    env: Optional[str] = None,
    backend: Optional[str] = None,
    model_name: Optional[str] = None
) -> DJIRAGPipeline:
    """
    便捷函数：创建生成流水线
    
    Args:
        config_path: 配置文件路径
        env: 环境名称
        backend: 强制指定LLM后端
        model_name: 强制指定模型名称
        
    Returns:
        DJIRAGPipeline实例
    """
    config = ConfigCenter(config_path=config_path, env=env)
    
    # 强制覆盖配置
    if backend:
        config.llm.backend = backend
    if model_name:
        config.llm.model_name = model_name
    
    return DJIRAGPipeline(config=config)


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DJI RAG 生成流水线")
    print("=" * 70)
    print("\n流水线阶段:")
    print("1. 查询理解与增强")
    print("2. 多路检索重排")
    print("3. 上下文筛选（文档ID去重、噪声过滤）")
    print("4. 上下文组装与提示词生成")
    print("5. LLM生成（支持vLLM/Ollama）")
    print("6. 草稿审查（幻觉防控校验）")
    print("7. 答案后处理（引用标记、格式美化、免责声明）")
    print("\n使用方式:")
    print("  from generation_pipeline import create_generation_pipeline")
    print("  pipeline = create_generation_pipeline(env='dev')")
    print("  result = pipeline.run('Mavic 4 Pro怎么起飞？')")
    print("  print(result.final_answer.text)")
    print("  print(pipeline.explain_result(result))")
