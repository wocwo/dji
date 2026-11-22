"""把手册知识库封装成 Agent 工具：manual_search / manual_fetch。

这是原 RAG 项目在 Agent 架构中的新位置：
不再是「一问一答的整条流水线」，而是 Agent 可以反复调用、组合、验证的一个工具。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from ..errors import KnowledgeBaseError
from ..knowledge import KnowledgeBase, Passage, extract_model_phrase, extract_terms
from ..types import ToolResult
from .base import Tool, ToolContext, ToolSpec


_GENERIC_TITLES = {"参数表", "表格", "文本", "无标题", "None", "null"}


def _matches_model(passage: Passage, model: str) -> bool:
    """软机型匹配：型号可能在 model/doc_id/title 任一处。"""
    if not model:
        return True
    haystack = f"{passage.model} {passage.doc_id} {passage.title}".lower()
    needle = model.lower().replace("dji ", "").strip()
    if needle and needle in haystack:
        return True
    parts = [part for part in needle.split() if part]
    return bool(parts) and all(part in haystack for part in parts)


def _focused_text(passage: Passage, query: str, max_chars: int) -> str:
    """长片段优先展示与查询词最相关的窗口，而不是简单截头。"""
    text = passage.text
    if len(text) <= max_chars:
        return text
    terms = extract_terms(query, max_terms=8)
    positions = [text.find(term) for term in terms if term and text.find(term) >= 0]
    if not positions:
        return passage.preview(max_chars)
    start = max(0, min(positions) - 160)
    end = min(len(text), start + max_chars)
    snippet = text[start:end]
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def _display_title(passage: Passage) -> str:
    """Chroma 中部分片段的 title 是“参数表”之类的泛化值，回退到 doc_id。"""
    title = passage.title
    if not title or title in _GENERIC_TITLES:
        doc_id = unquote(passage.doc_id).strip()
        return doc_id or passage.chunk_id
    return title


def _display_section(passage: Passage) -> str:
    section = passage.section
    if not section:
        return ""
    # Chroma 的部分 section 实际是正文句子，过长的不要塞进引用标签
    if len(section) > 30 or section.endswith(("。", "；", "！", "？")):
        return ""
    return section


def _format_passage_header(passage: Passage, ref: str) -> str:
    parts = [f"[{ref}]"]
    label = _display_title(passage)
    if label:
        parts.append(label)
    if passage.page not in (None, "", 0):
        parts.append(f"第{passage.page}页")
    section = _display_section(passage)
    if section:
        parts.append(section)
    if passage.model and passage.model not in ("DJI Device", "未知"):
        parts.append(passage.model)
    return " · ".join(parts)


class ManualSearchTool(Tool):
    """手册检索工具。"""

    def __init__(self, kb: KnowledgeBase, *, default_top_k: int = 4, max_passage_chars: int = 700):
        self.kb = kb
        self.default_top_k = default_top_k
        self.max_passage_chars = max_passage_chars

    spec = ToolSpec(
        name="manual_search",
        description=(
            "检索大疆官方产品手册知识库。返回若干带引用编号的片段（[S1]、[S2]…）。"
            "当问题涉及产品规格、操作步骤、固件升级、故障排查、电池/桨叶/遥控器等手册事实时，"
            "必须先调用本工具；可以多次调用，每次换更具体的关键词或机型。"
            "如果结果为空，尝试缩短关键词或改用型号名。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索词，建议包含机型+主题，例如「Mavic 4 Pro 智能飞行电池 保养」",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数，2-8，默认 4",
                    "default": 4,
                },
                "model": {
                    "type": "string",
                    "description": "按机型过滤（仅当语料带机型元数据时有效，可选）",
                },
            },
            "required": ["query"],
        },
        risk="safe",
        tags=["knowledge", "rag"],
    )

    def run(
        self,
        ctx: ToolContext,
        query: str,
        top_k: int = 0,
        model: Optional[str] = None,
    ) -> ToolResult:
        if ctx.kb is None:
            return ToolResult(
                ok=False,
                content="知识库未配置，无法检索手册。请检查 knowledge.backend / 数据目录。",
                error="knowledge base unavailable",
            )
        query = str(query or "").strip()
        if not query:
            return ToolResult(ok=False, content="检索词为空", error="empty query")

        k = int(top_k or self.default_top_k)
        k = max(1, min(k, 10))
        model = (
            model
            or getattr(ctx, "requested_model", "")
            or extract_model_phrase(query)
        )
        start = time.time()
        try:
            passages = ctx.kb.search(query, top_k=k, model=model or None)
        except KnowledgeBaseError as exc:
            return ToolResult(
                ok=False,
                content=f"手册检索失败：{exc}",
                error=str(exc),
                data={"query": query, "backend": getattr(ctx.kb, "name", "")},
            )

        if model and passages:
            scoped = [passage for passage in passages if _matches_model(passage, model)]
            if scoped:
                passages = scoped

        elapsed_ms = (time.time() - start) * 1000
        backend = getattr(ctx.kb, "last_backend", getattr(ctx.kb, "name", ""))
        if not passages:
            return ToolResult(
                ok=True,
                content=(
                    f"未检索到与「{query}」相关的手册片段。\n"
                    "建议：换用更短的的关键词、补充具体机型，或改用英文型号名。"
                ),
                data={"query": query, "count": 0, "backend": backend},
            )

        citations = []
        blocks: List[str] = [
            f"检索到 {len(passages)} 条片段（knowledge backend={backend}，{elapsed_ms:.0f}ms）：",
            "",
        ]
        for passage in passages:
            ref = ctx.cite(
                passage.chunk_id,
                doc_id=passage.doc_id,
                title=_display_title(passage),
                section=_display_section(passage),
                page=passage.page,
                model=passage.model,
                version=str(passage.metadata.get("version") or ""),
                doc_type=str(passage.metadata.get("doc_type") or ""),
                source=passage.source,
            )
            citation = ctx.citations[passage.chunk_id] if passage.chunk_id in ctx.citations else None
            if citation is not None:
                citations.append(citation)
            blocks.append(_format_passage_header(passage, ref))
            blocks.append(_focused_text(passage, query, self.max_passage_chars))
            blocks.append("")

        return ToolResult(
            ok=True,
            content="\n".join(blocks).strip(),
            data={
                "query": query,
                "count": len(passages),
                "backend": backend,
                "requested_model": model,
                "passages": [
                    {
                        "chunk_id": p.chunk_id,
                        "score": round(p.score, 4),
                        "title": p.title,
                        "doc_id": p.doc_id,
                        "page": p.page,
                        "section": p.section,
                        "model": p.model,
                        "version": p.metadata.get("version", ""),
                        "doc_type": p.metadata.get("doc_type", ""),
                        "type": p.metadata.get("type", ""),
                        "table_id": p.metadata.get("table_id", ""),
                        "preview": p.preview(240),
                        # 供 grounding 校验逐句比对使用
                        "text": p.text[:4000],
                    }
                    for p in passages
                ],
            },
            citations=citations,
        )


class ManualFetchTool(Tool):
    """按 chunk_id 取回完整片段。"""

    def __init__(self, kb: KnowledgeBase, *, max_chars: int = 3000):
        self.kb = kb
        self.max_chars = max_chars

    spec = ToolSpec(
        name="manual_fetch",
        description=(
            "根据 manual_search 结果中的 chunk_id 取回完整片段。"
            "当搜索结果被截断、需要完整步骤或参数表时使用。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "manual_search 结果中的 chunk_id"},
            },
            "required": ["chunk_id"],
        },
        risk="safe",
        tags=["knowledge", "rag"],
    )

    def run(self, ctx: ToolContext, chunk_id: str) -> ToolResult:
        if ctx.kb is None:
            return ToolResult(ok=False, content="知识库未配置", error="knowledge base unavailable")
        chunk_id = str(chunk_id or "").strip()
        if not chunk_id:
            return ToolResult(ok=False, content="chunk_id 为空", error="empty chunk_id")
        try:
            passage = ctx.kb.fetch(chunk_id)
        except KnowledgeBaseError as exc:
            return ToolResult(ok=False, content=f"取回片段失败：{exc}", error=str(exc))
        if passage is None:
            return ToolResult(
                ok=False,
                content=f"未找到 chunk_id={chunk_id} 的片段。",
                error="chunk not found",
            )
        ref = ctx.cite(
            passage.chunk_id,
            doc_id=passage.doc_id,
            title=_display_title(passage),
            section=_display_section(passage),
            page=passage.page,
            model=passage.model,
            version=str(passage.metadata.get("version") or ""),
            doc_type=str(passage.metadata.get("doc_type") or ""),
            source=passage.source,
        )
        text = passage.text
        truncated = len(text) > self.max_chars
        if truncated:
            text = text[: self.max_chars] + "…（片段已截断）"
        content = f"{_format_passage_header(passage, ref)}\n{text}"
        return ToolResult(
            ok=True,
            content=content,
            data={
                "chunk_id": passage.chunk_id,
                "doc_id": passage.doc_id,
                "title": passage.title,
                "page": passage.page,
                "truncated": truncated,
                "text": passage.text[:4000],
            },
            citations=[ctx.citations[passage.chunk_id]] if passage.chunk_id in ctx.citations else [],
            truncated=truncated,
        )


__all__ = ["ManualSearchTool", "ManualFetchTool"]
