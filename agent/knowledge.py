"""知识库：把原 RAG 项目的手册数据变成 Agent 的检索工具后端。

提供两类实现：
- :class:`ChromaKnowledgeBase`：直接读现有 ``chroma_final`` 集合，用 Chroma 的
  ``where_document={"$contains": ...}`` 做**词法检索**。原因：原配置使用
  ``precomputed`` 嵌入，查询文本无法实时向量化，语义通道实际是随机向量；
  词法检索在离线环境下更可靠，也无需加载 2.6GB 的 npz。
- :class:`JsonlKnowledgeBase`：小型 JSONL 语料的 BM25 检索，用于离线演示、
  CI 与无向量库环境。

两者都输出统一的 :class:`Passage`，由工具层负责格式化为 observation 与引用。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import KnowledgeSettings
from .errors import KnowledgeBaseError
from .cross_encoder_reranker import default_reranker
from .page_reranker import PageReranker


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass
class Passage:
    """一条检索到的知识片段。"""

    chunk_id: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

    @property
    def title(self) -> str:
        title = str(self.metadata.get("title") or "").strip()
        return "" if title in ("无标题", "None", "null") else title

    @property
    def section(self) -> str:
        return str(self.metadata.get("section") or "").strip()

    @property
    def doc_id(self) -> str:
        return str(self.metadata.get("doc_id") or "")

    @property
    def page(self) -> Any:
        page = self.metadata.get("page")
        if isinstance(page, str) and page.strip().isdigit():
            return int(page.strip())
        return page

    @property
    def model(self) -> str:
        return str(self.metadata.get("model") or "")

    def preview(self, max_chars: int = 200) -> str:
        text = re.sub(r"\s+", " ", self.text).strip()
        return text if len(text) <= max_chars else text[:max_chars] + "…"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata,
            "source": self.source,
        }


class KnowledgeBase(ABC):
    """知识库接口。"""

    name: str = "unknown"
    description: str = ""

    @abstractmethod
    def search(self, query: str, top_k: int = 4, model: Optional[str] = None) -> List[Passage]:
        ...

    def fetch(self, chunk_id: str) -> Optional[Passage]:
        return None

    def stats(self) -> Dict[str, Any]:
        return {"backend": self.name, "description": self.description}

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 查询词提取
# ---------------------------------------------------------------------------

_ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_\./]{1,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_QUESTION_WORDS = [
    "请问", "为什么", "为何", "怎么办", "怎么样", "怎么", "如何", "什么",
    "哪些", "哪个", "是否", "可以", "能否", "需要", "应该", "一下",
    "帮我", "我想", "我想知道", "告诉", "介绍", "说明", "吗", "呢", "吧",
]
_GENERIC_TERMS = {
    "大疆", "无人机", "飞行器", "产品", "手册", "说明书", "用户", "使用",
    "问题", "情况", "方法", "操作", "功能", "支持", "相关", "进行", "以及",
    "无法", "多少", "多久", "怎样", "这个", "那个", "我们", "他们",
}

# 领域词表：解决“智能飞行电池低温下”这类无空格长串的切词。
# 按长度倒序匹配，长词优先（“智能飞行电池” 优于 “电池”）。
_TOPIC_LEXICON = sorted(
    {
        "智能飞行电池", "飞行电池", "电池保养", "低温环境", "固件升级", "固件回退",
        "遥控器对频", "指南针校准", "飞行限制", "智能跟随", "存储卡", "避障",
        "图传", "返航", "起飞", "降落", "桨叶", "电机", "充电", "续航",
        "保养", "低温", "固件", "升级", "对频", "遥控器", "校准", "激活",
        "限高", "禁飞", "云台", "相机", "拍照", "录像", "安全", "警告",
        "参数", "规格", "电池",
    },
    key=lambda term: (-len(term), term),
)

_MODEL_PATTERNS = [
    re.compile(r"Mavic\s*\d+\s*(?:Pro|Mini|Enterprise|Cine|Classic)?", re.IGNORECASE),
    re.compile(r"Mini\s*\d+\s*(?:Pro|SE)?", re.IGNORECASE),
    re.compile(r"Air\s*\d+\s*S?", re.IGNORECASE),
    re.compile(r"Matrice\s*\d+\s*[A-Za-z]*", re.IGNORECASE),
    re.compile(r"Avata\s*\d*", re.IGNORECASE),
    re.compile(r"Phantom\s*\d+", re.IGNORECASE),
    re.compile(r"Inspire\s*\d+", re.IGNORECASE),
    re.compile(r"\b(?:Neo|Flip)\b", re.IGNORECASE),
]


def extract_model_phrase(query: str) -> Optional[str]:
    """从查询中提取机型短语（如 Mavic 4 Pro），用于定位对应手册。"""
    for pattern in _MODEL_PATTERNS:
        match = pattern.search(query or "")
        if match:
            raw = re.sub(r"\s+", " ", match.group(0)).strip()
            # 统一大小写，避免 $contains 因 "mini 4 pro" 匹配不到 "Mini 4 Pro"
            parts = []
            for part in raw.split(" "):
                if part.isalpha():
                    parts.append(part[:1].upper() + part[1:].lower() if len(part) > 1 else part.upper())
                else:
                    parts.append(part)
            return " ".join(parts)
    return None


def _model_slug(phrase: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")


def _match_lexicon(query: str) -> List[str]:
    found: List[str] = []
    for term in _TOPIC_LEXICON:
        if term in query and not any(term in existing for existing in found):
            found.append(term)
    return found


def extract_terms(query: str, max_terms: int = 5) -> List[str]:
    """从自然语言查询中提取适合 ``$contains`` 的检索词。

    顺序：英文型号 → 领域词表（长词优先）→ 剩余名词短语 → 子串兜底。
    领域词表能可靠拆出「智能飞行电池 / 低温 / 保养」，
    避免把「智能飞行电池低温」当成一个不存在的长词去检索。
    """
    query = (query or "").strip()
    if not query:
        return []

    terms: List[str] = []

    def add(bucket: List[str], term: str) -> None:
        term = term.strip()
        if len(term) < 2 or term in _GENERIC_TERMS:
            return
        if any(term == existing or term in existing for existing in bucket):
            return
        bucket.append(term)

    # 1) 英文/数字型号优先
    ascii_terms: List[str] = []
    for match in _ASCII_TERM_RE.finditer(query):
        term = match.group(0).strip("-_./")
        if len(term) >= 2:
            add(ascii_terms, term)

    # 2) 领域词表
    lex_terms = _match_lexicon(query)

    # 3) 按疑问词 / 介词切出剩余短语
    stop_pattern = "|".join(re.escape(w) for w in _QUESTION_WORDS)
    segments: List[str] = []
    for run in _CJK_RUN_RE.findall(query):
        for segment in re.split(stop_pattern, run):
            segment = segment.strip("的了吗呢吧啊")
            if len(segment) < 2:
                continue
            for piece in re.split(r"[在于的]", segment):
                piece = piece.strip("与和及或")
                while len(piece) > 3 and piece[-1] in "下中时后前里内外上":
                    piece = piece[:-1]
                if len(piece) >= 2:
                    segments.append(piece)

    for term in ascii_terms:
        add(terms, term)
    for term in lex_terms:
        add(terms, term)

    for piece in segments:
        if not lex_terms:
            add(terms, piece)
            continue
        residual = piece
        for lex in lex_terms:
            residual = residual.replace(lex, " ")
        for rest in residual.split():
            add(terms, rest)

    # 4) 仍有余量时，用子串补召回（长词的 2/4 字窗口）
    if len(terms) < max_terms:
        secondary: List[str] = []
        for term in lex_terms:
            if len(term) >= 4:
                add(secondary, term[:4])
                add(secondary, term[-4:])
                add(secondary, term[:2])
                add(secondary, term[-2:])
        for piece in segments:
            if lex_terms and any(lex in piece for lex in lex_terms):
                continue
            if len(piece) >= 4:
                add(secondary, piece[:4])
                add(secondary, piece[-4:])
            add(secondary, piece[:2])
            add(secondary, piece[-2:])
        for term in secondary:
            add(terms, term)
            if len(terms) >= max_terms:
                break

    return terms[:max_terms]


def clean_retrieved_text(raw: str) -> str:
    """清理 PDF 抽取产生的表格边框/空白，提升 observation 可读性。"""
    text = (raw or "").replace("\x00", " ")
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[|\s:\-\d]+", line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r"[\s:\-\d]+", c)]
            if not cells:
                continue
            line = " ".join(cells)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chroma 词法检索
# ---------------------------------------------------------------------------

class ChromaKnowledgeBase(KnowledgeBase):
    """基于现有 Chroma 集合的词法检索。"""

    def __init__(
        self,
        persist_path: str = "./chroma_final",
        collection_name: str = "uav_manual_db",
        *,
        candidate_limit: int = 300,
        max_terms: int = 5,
        timeout: float = 25.0,
        cache_size: int = 128,
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.candidate_limit = max(20, int(candidate_limit))
        self.max_terms = max(1, int(max_terms))
        self.timeout = timeout
        self.cache_size = cache_size
        self.name = "chroma-lexical"
        self.description = (
            f"大疆官方手册库（本地 Chroma 词法检索，集合 {collection_name}）"
        )
        self._collection: Any = None
        self._count: Optional[int] = None
        self._open_error: Optional[str] = None
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cache: "OrderedDict[Tuple[str, Optional[str], int], List[Passage]]" = OrderedDict()
        self._doc_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        self._page_cache: "OrderedDict[str, Passage]" = OrderedDict()

    # -- 初始化 -----------------------------------------------------------
    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        with self._lock:
            if self._collection is not None:
                return self._collection
            path = Path(self.persist_path).expanduser()
            if not path.exists():
                raise KnowledgeBaseError(f"Chroma 目录不存在: {path}")
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(path))
                self._collection = client.get_collection(self.collection_name)
                self._count = int(self._collection.count())
            except Exception as exc:
                self._open_error = f"{type(exc).__name__}: {exc}"
                raise KnowledgeBaseError(
                    f"打开 Chroma 集合失败（{path} / {self.collection_name}）: {self._open_error}"
                ) from exc
            return self._collection

    # -- 检索 -------------------------------------------------------------
    def search(self, query: str, top_k: int = 4, model: Optional[str] = None) -> List[Passage]:
        query = (query or "").strip()
        if not query:
            return []
        key = (query, model, int(top_k))
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        terms = extract_terms(query, self.max_terms)
        if not terms:
            return []

        collection = self._ensure_collection()
        batches = self._query_terms(collection, terms, model)
        # 机型定位：先找到对应手册，再在该文档内检索（分块常把“低温”与“电池”拆开）
        try:
            expanded = self._expand_model_docs(collection, query, model)
            if expanded:
                batches.append(expanded)
        except Exception:
            pass
        passages = self._merge_and_score(query, terms, batches)
        passages = passages[: max(1, int(top_k))]

        with self._cache_lock:
            self._cache[key] = passages
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return passages

    def _query_one(
        self,
        collection: Any,
        term: str,
        model: Optional[str],
        second: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        where_document: Dict[str, Any] = {"$contains": term}
        limit = self.candidate_limit
        if second:
            where_document = {"$and": [{"$contains": term}, {"$contains": second}]}
            limit = min(self.candidate_limit, 120)
        kwargs: Dict[str, Any] = {
            "where_document": where_document,
            "limit": limit,
            "include": ["documents", "metadatas"],
        }
        if model:
            kwargs["where"] = {"model": model}
        result = collection.get(**kwargs)
        if model and not result.get("ids"):
            # 语料里的 model 可能是 "DJI Mavic 4 Pro" 而查询里是 "Mavic 4 Pro"；
            # 精确 where 查不到时回退到无过滤，再由工具层做软机型匹配。
            kwargs.pop("where", None)
            result = collection.get(**kwargs)
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        rows: List[Dict[str, Any]] = []
        for idx, chunk_id in enumerate(ids):
            text = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) else {}
            if not text:
                continue
            text = clean_retrieved_text(text)
            if not text:
                continue
            rows.append({"id": str(chunk_id), "text": text, "metadata": meta or {}})
        return rows

    def _expand_model_docs(
        self,
        collection: Any,
        query: str,
        model: Optional[str],
    ) -> List[Dict[str, Any]]:
        """先定位机型手册，再取回该文档的片段，弥补 chunk 级检索拆词问题。"""
        phrase = model or extract_model_phrase(query)
        if not phrase:
            return []
        slug = _model_slug(phrase)
        if len(slug) < 3:
            return []

        # 用一个高区分度主题词与机型短语做交集，尽量稳定地拿到 doc_id
        probe_terms = ["电池", "固件", "遥控器", "保养", "参数"]
        doc_ids: List[str] = []
        probes = [collection.get(where_document={"$contains": phrase}, limit=20, include=["metadatas"])]
        for topic in probe_terms:
            probes.append(
                collection.get(
                    where_document={"$and": [{"$contains": phrase}, {"$contains": topic}]},
                    limit=10,
                    include=["metadatas"],
                )
            )
        for probe in probes:
            for meta in probe.get("metadatas") or []:
                doc_id = str((meta or {}).get("doc_id") or "")
                if doc_id and slug in doc_id.lower() and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
        # 用户手册优先于配件说明
        doc_ids.sort(key=lambda item: ("user_manual" not in item.lower(), len(item)))

        rows: List[Dict[str, Any]] = []
        for doc_id in doc_ids[:2]:
            with self._cache_lock:
                cached = self._doc_cache.get(doc_id)
                if cached is not None:
                    self._doc_cache.move_to_end(doc_id)
            if cached is None:
                result = collection.get(
                    where={"doc_id": doc_id},
                    limit=6000,
                    include=["documents", "metadatas"],
                )
                # 按页合并碎片：低温注意事项常被拆成“标题 + 多条 bullet”
                pages: "OrderedDict[Any, List[Dict[str, Any]]]" = OrderedDict()
                for chunk_id, document, metadata in zip(
                    result.get("ids") or [],
                    result.get("documents") or [],
                    result.get("metadatas") or [],
                ):
                    text = clean_retrieved_text(document or "")
                    if not text:
                        continue
                    metadata = metadata or {}
                    page = metadata.get("page")
                    if isinstance(page, str) and page.strip().isdigit():
                        page = int(page.strip())
                    pages.setdefault(page, []).append(
                        {"id": str(chunk_id), "text": text, "metadata": metadata}
                    )
                cached = []
                for page, items in pages.items():
                    merged_text = "\n".join(item["text"] for item in items)
                    # 跳过封面/目录页，避免把 TOC 当成答案
                    if page in (0, 1):
                        continue
                    if len(re.findall(r"\.{3,}\s*\d+", merged_text)) > 3:
                        continue
                    if len(merged_text.strip()) < 30:
                        continue
                    synthetic_id = f"{doc_id}__p{page}"
                    metadata = dict(items[0]["metadata"])
                    metadata["page"] = page
                    cached.append(
                        {
                            "id": synthetic_id,
                            "text": merged_text,
                            "metadata": metadata,
                            "expanded": True,
                        }
                    )
                    with self._cache_lock:
                        self._page_cache[synthetic_id] = Passage(
                            chunk_id=synthetic_id,
                            text=merged_text,
                            metadata=metadata,
                            source=f"chroma:{self.collection_name}",
                        )
                        while len(self._page_cache) > 2000:
                            self._page_cache.popitem(last=False)
                with self._cache_lock:
                    self._doc_cache[doc_id] = cached
                    self._doc_cache.move_to_end(doc_id)
                    while len(self._doc_cache) > 4:
                        self._doc_cache.popitem(last=False)
            rows.extend(cached)
        return rows

    def _query_terms(self, collection: Any, terms: Sequence[str], model: Optional[str]) -> List[List[Dict[str, Any]]]:
        """并行检索各词，并对关键中文词做 `$and` 组合检索。

        单纯 `$contains` + limit 会返回“前 N 条任意匹配”，常见词很容易淹没长尾文档；
        对两个主题词做交集能显著提升候选质量。旧版 Chroma 不支持 `$and` 时自动忽略组合查询。
        """
        cjk_terms = [term for term in terms if not _ASCII_TERM_RE.fullmatch(term)]
        pairs: List[Tuple[str, str]] = []
        if len(cjk_terms) >= 2:
            pairs.append((cjk_terms[0], cjk_terms[1]))
        if len(cjk_terms) >= 3:
            pairs.append((cjk_terms[0], cjk_terms[2]))
            pairs.append((cjk_terms[1], cjk_terms[2]))

        jobs: List[Tuple[str, Optional[str]]] = [(term, None) for term in terms] + pairs
        results: List[List[Dict[str, Any]]] = []
        workers = min(4, len(jobs))
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kb-chroma") as pool:
                future_map = {
                    pool.submit(self._query_one, collection, first, model, second): (first, second)
                    for first, second in jobs
                }
                for future in as_completed(future_map, timeout=self.timeout):
                    first, second = future_map[future]
                    try:
                        results.append(future.result())
                    except Exception:
                        # 组合查询失败（例如旧版 Chroma 不支持 $and）不影响单词检索
                        if second is None:
                            raise
            return results
        except FuturesTimeout as exc:
            raise KnowledgeBaseError(f"Chroma 检索超时（>{self.timeout:.0f}s）") from exc
        except KnowledgeBaseError:
            raise
        except Exception as exc:
            # 某些 Chroma 版本对并发读取不友好，串行重试一次
            results = []
            for first, second in jobs:
                try:
                    results.append(self._query_one(collection, first, model, second))
                except Exception as inner:
                    if second is None:
                        raise KnowledgeBaseError(f"Chroma 检索失败: {inner}") from inner
            return results

    def _merge_and_score(
        self,
        query: str,
        terms: Sequence[str],
        batches: Sequence[Sequence[Dict[str, Any]]],
    ) -> List[Passage]:
        merged: Dict[str, Dict[str, Any]] = {}
        for batch in batches:
            for row in batch:
                merged.setdefault(row["id"], row)

        # 候选集内计算 IDF：常见词（智能飞行电池/保养）降权，稀有主题词（低温环境）升权
        doc_freq: Dict[str, int] = {}
        for term in terms:
            lowered = term.lower()
            doc_freq[term] = sum(
                1 for row in merged.values() if lowered in str(row.get("text", "")).lower()
            )
        candidate_count = max(len(merged), 1)
        weights: List[float] = []
        for term in terms:
            df = doc_freq[term]
            if df <= 0:
                weights.append(0.0)
                continue
            base = (
                (0.35 if len(term) <= 4 else 0.5)
                if _ASCII_TERM_RE.fullmatch(term)
                else (1.0 if len(term) >= 4 else 0.7)
            )
            idf = math.log((candidate_count + 1) / (df + 1)) + 0.5
            weights.append(base * idf)
        total_weight = sum(weights) or 1.0

        query_lower = query.lower().strip()
        scored: List[Passage] = []
        seen_text: set[str] = set()
        for chunk_id, row in merged.items():
            text = str(row["text"] or "")
            if not text.strip():
                continue
            fingerprint = re.sub(r"\s+", "", text)[:160]
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)

            lower = text.lower()
            coverage = sum(
                weight for term, weight in zip(terms, weights) if term.lower() in lower
            ) / total_weight
            hit_score = min(
                sum(min(lower.count(term.lower()), 5) * weight for term, weight in zip(terms, weights))
                / (total_weight * 2),
                1.0,
            )
            meta = row.get("metadata") or {}
            meta_text = " ".join(
                str(meta.get(key, "")) for key in ("title", "section", "keywords", "model")
            ).lower()
            meta_score = sum(
                weight for term, weight in zip(terms, weights) if term.lower() in meta_text
            ) / total_weight
            phrase_score = 1.0 if query_lower and query_lower in lower else 0.0
            score = 0.55 * coverage + 0.25 * hit_score + 0.10 * meta_score + 0.10 * phrase_score
            # 轻微偏好信息密度高的片段，避免只命中标题的短块
            score += min(len(text) / 4000, 0.03)
            # 机型定位命中的文档片段加权（用户明确指定机型时）
            if row.get("expanded"):
                score += 0.08

            scored.append(
                Passage(
                    chunk_id=chunk_id,
                    text=text,
                    score=round(min(score, 1.0), 6),
                    metadata=meta,
                    source=f"chroma:{self.collection_name}",
                )
            )

        scored.sort(key=lambda p: (p.score, -len(p.text), p.chunk_id), reverse=True)
        return scored

    def fetch(self, chunk_id: str) -> Optional[Passage]:
        if not chunk_id:
            return None
        # 机型扩展检索产生的“整页合成片段”
        with self._cache_lock:
            cached_page = self._page_cache.get(chunk_id)
            if cached_page is not None:
                return cached_page
        collection = self._ensure_collection()
        result = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        ids = result.get("ids") or []
        if not ids:
            return None
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return Passage(
            chunk_id=str(ids[0]),
            text=clean_retrieved_text(str(documents[0] if documents else "")),
            metadata=dict(metadatas[0] or {}) if metadatas else {},
            source=f"chroma:{self.collection_name}",
        )

    def stats(self) -> Dict[str, Any]:
        count = None
        try:
            self._ensure_collection()
            count = self._count
        except KnowledgeBaseError:
            count = None
        return {
            "backend": self.name,
            "path": self.persist_path,
            "collection": self.collection_name,
            "count": count,
            "open_error": self._open_error,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# JSONL BM25（离线演示 / 小语料）
# ---------------------------------------------------------------------------

_TOKEN_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9\-_\./]*")
_TOKEN_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """轻量分词：英文/数字保留原词，中文用二元组。"""
    text = (text or "").lower()
    tokens = _TOKEN_ASCII_RE.findall(text)
    for run in _TOKEN_CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class JsonlKnowledgeBase(KnowledgeBase):
    """JSONL/JSON 小语料的 BM25 检索。"""

    def __init__(
        self,
        path: str,
        *,
        max_docs: int = 50000,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.path = path
        self.max_docs = max_docs
        self.k1 = k1
        self.b = b
        self.name = "jsonl-bm25"
        self.description = f"页面级手册语料（BM25 词法检索，{path}）"
        self._passages: List[Passage] = []
        self._tokens: List[List[str]] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        self._loaded = False
        self._lock = threading.Lock()
        self._reranker = PageReranker(pool=30)
        self._cross_encoder = default_reranker()
        self.rerank_calls = 0
        self.rerank_skipped = 0

    # -- 加载 -------------------------------------------------------------
    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            path = Path(self.path).expanduser()
            if not path.exists():
                raise KnowledgeBaseError(f"JSONL 语料不存在: {path}")

            raw_items: List[Dict[str, Any]] = []
            text = path.read_text(encoding="utf-8")
            stripped = text.lstrip()
            if stripped.startswith("["):
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        raw_items = [x for x in data if isinstance(x, dict)]
                except json.JSONDecodeError as exc:
                    raise KnowledgeBaseError(f"JSON 语料解析失败: {exc}") from exc
            else:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise KnowledgeBaseError(f"JSONL 第 {line_no} 行解析失败: {exc}") from exc
                    if isinstance(item, dict):
                        raw_items.append(item)
                    if len(raw_items) >= self.max_docs:
                        break

            passages: List[Passage] = []
            for idx, item in enumerate(raw_items[: self.max_docs]):
                body = item.get("text") or item.get("document") or item.get("content") or ""
                if not isinstance(body, str) or not body.strip():
                    continue
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                merged_meta = dict(metadata)
                for key in (
                    "doc_id", "title", "section", "page", "model", "type", "lang",
                    "keywords", "version", "doc_type", "source", "source_kind",
                ):
                    if key in item and key not in merged_meta:
                        merged_meta[key] = item[key]
                chunk_id = str(item.get("chunk_id") or item.get("id") or f"{merged_meta.get('doc_id', 'doc')}-{idx}")
                passages.append(
                    Passage(
                        chunk_id=chunk_id,
                        text=body,
                        metadata=merged_meta,
                        source=str(item.get("source") or f"jsonl:{path.name}"),
                    )
                )
            if not passages:
                raise KnowledgeBaseError(f"语料为空或字段不兼容: {path}")

            self._passages = passages
            self._tokens = [tokenize(p.text) for p in passages]
            self._df = Counter()
            for tokens in self._tokens:
                self._df.update(set(tokens))
            total = sum(len(t) for t in self._tokens)
            self._avgdl = total / len(self._tokens) if self._tokens else 0.0
            self._loaded = True

    def _needs_rerank(self, query: str, scored: Sequence[Tuple[float, Passage]]) -> bool:
        """只有低置信度候选才值得跑 cross-encoder，避免每次多 1–2 秒。"""
        if len(scored) < 2:
            return False
        (s1, p1), (s2, _p2) = scored[0], scored[1]
        if s1 <= 0:
            return False
        gap = (s1 - s2) / s1
        terms = [
            term for term in extract_terms(query, max_terms=8)
            if len(term) >= 2 and not term.isdigit()
        ]
        text = (p1.text or "")[:2000]
        coverage = (
            sum(1 for term in terms if term in text) / len(terms)
            if terms else 1.0
        )
        strong = gap >= 0.22 and coverage >= 0.45 and str(p1.page) != "0"
        return not strong

    # -- 检索 -------------------------------------------------------------
    def search(self, query: str, top_k: int = 4, model: Optional[str] = None) -> List[Passage]:
        query = (query or "").strip()
        if not query:
            return []
        self._load()
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        n_docs = len(self._passages)
        avgdl = self._avgdl or 1.0
        model_phrase = model or extract_model_phrase(query)
        model_slug = _model_slug(model_phrase) if model_phrase else ""
        query_token_set = set(q_tokens)
        scored: List[Tuple[float, Passage]] = []

        for passage, tokens in zip(self._passages, self._tokens):
            exact_model = True
            if model and passage.model:
                requested = model.lower().strip()
                actual = passage.model.lower()
                if requested not in actual:
                    # 允许系列级兜底："Mavic 3 Pro" 找不到精确页面时，
                    # 仍可检索 "DJI Mavic 3" 系列页，但后面会降低权重。
                    core = requested.removeprefix("dji ").strip()
                    family = " ".join(core.split()[:2])
                    if not family or family not in actual:
                        continue
                    exact_model = False
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for token in q_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                df = self._df.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (freq * (self.k1 + 1)) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
            if score <= 0:
                continue
            if not exact_model:
                # 同系列兜底可以召回，但不能压过精确机型页面
                score *= 0.5
            # 查询词覆盖率：避免只因为型号名重复就排到前面
            token_set = set(tokens)
            coverage = len(query_token_set & token_set) / len(query_token_set)
            score *= 1.0 + 0.6 * coverage
            # 内容词覆盖率：型号词不能主导排序，低温/电池/校准这类词才是主题
            content_terms = [
                term for term in extract_terms(query, max_terms=8)
                if term
                and term.lower() not in (model_phrase or "").lower()
                and not term.isdigit()
                and term not in ("怎么", "如何", "什么", "这个", "一下")
            ]
            if content_terms:
                content_hits = sum(1 for term in content_terms if term in passage.text)
                score *= 1.0 + 1.5 * (content_hits / len(content_terms))
                # 长主题词（低温/校准/固件）命中比型号词更有区分度
                for term in content_terms:
                    if len(term) >= 2 and term in passage.text:
                        score *= 1.35
                    if len(term) >= 2 and term in f"{passage.section} {passage.title}":
                        score *= 1.12
                # 温度类问题里，“低温/高温”是强主题词，不能被电池/型号词淹没
                for topic in ("低温", "高温", "校准", "固件", "返航", "避障"):
                    if topic in query and topic in passage.text:
                        score *= 1.8
            if str(passage.page) == "0":
                # 旧 chunk 的 page=0 常是目录/封面，降权
                score *= 0.75
            # 规格/参数页：查询问“范围/规格/参数”时优先技术规格章节
            q_norm = query.replace("环境", "")
            t_norm = passage.text.replace("环境", "")
            if "充电温度" in q_norm and "充电温度" in t_norm:
                score *= 1.5
            if "温度范围" in q_norm and t_norm.count("℃") >= 2:
                score *= 1.3
            # 视觉/返航定义页：避免 “视觉/返航” 只命中目录或发布说明
            if "视觉系统" in query and "视觉系统" in passage.text:
                if any(word in passage.text for word in ("摄像头", "感知", "视觉定位")):
                    score *= 1.5
                if any(word in passage.text for word in ("摄像头", "位于", "组成")):
                    score *= 1.8
                if "分别位于" in passage.text:
                    score *= 1.4
            if "全向" in query and "全向" in passage.text and "视觉" in passage.text:
                score *= 1.3
                if "摄像头" in passage.text and any(word in passage.text for word in ("前、后、侧", "方向", "位于")):
                    score *= 1.8
            if "返航点" in query and "返航点" in passage.text:
                if "GNSS" in passage.text or "记录" in passage.text:
                    score *= 1.5
                if "记录" in query and "记录" in passage.text and "GNSS" in passage.text:
                    score *= 1.8
            section_title = f"{passage.section} {passage.title}"
            if any(word in query for word in ("规格", "参数", "尺寸", "重量")):
                if any(word in section_title for word in ("规格", "参数", "技术规格")):
                    score *= 1.4
            if passage.metadata.get("type") == "table":
                # 表格是针对参数/规格问题的高密度证据
                score *= 1.10
            # 查询里明确机型时，优先对应手册；避免“Mavic 4 Pro”问题被 Mini 手册淹没
            if model_slug:
                doc_identity = f"{passage.doc_id} {passage.model}".lower()
                if model_slug in doc_identity:
                    score *= 1.45
                elif model_phrase and model_phrase.lower() in passage.text.lower():
                    score *= 1.2
                else:
                    score *= 0.9
            # PDF 逐页精读的页面优先于旧 chunk 抢救页
            if passage.metadata.get("source_kind") == "pdf":
                score *= 1.15
            # 标题/章节命中加成
            meta_text = f"{passage.title} {passage.section} {passage.metadata.get('keywords', '')}".lower()
            for token in set(q_tokens):
                if len(token) >= 2 and token in meta_text:
                    score *= 1.05
                    break
            scored.append((score, passage))

        scored.sort(key=lambda item: (item[0], item[1].chunk_id), reverse=True)
        if self._cross_encoder is not None:
            import os
            mode = os.environ.get("AGENT_RERANK_MODE", "auto").lower()
            if mode == "off" or (mode == "auto" and not self._needs_rerank(query, scored)):
                scored = scored[: max(1, int(top_k))]
                self.rerank_skipped += 1
            else:
                pool = int(os.environ.get("AGENT_RERANK_POOL", "18"))
                scored = self._cross_encoder.rank(query, scored[:pool], top_k=max(1, int(top_k)))
                self.rerank_calls += 1
        else:
            scored = self._reranker.rank(query, model_phrase or model or "", scored)
        results: List[Passage] = []
        for score, passage in scored[: max(1, int(top_k))]:
            # 归一化到 0-1 便于跨后端展示
            normalized = score / (score + 5.0) if score > 0 else 0.0
            results.append(
                Passage(
                    chunk_id=passage.chunk_id,
                    text=passage.text,
                    score=round(normalized, 4),
                    metadata=dict(passage.metadata),
                    source=passage.source,
                )
            )
        return results

    def fetch(self, chunk_id: str) -> Optional[Passage]:
        self._load()
        for passage in self._passages:
            if passage.chunk_id == chunk_id:
                return passage
        return None

    def stats(self) -> Dict[str, Any]:
        try:
            self._load()
            count = len(self._passages)
        except KnowledgeBaseError:
            count = None
        return {
            "backend": self.name,
            "path": self.path,
            "count": count,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# 自动选择 + 降级
# ---------------------------------------------------------------------------

class AutoKnowledgeBase(KnowledgeBase):
    """优先使用主后端，失败时降级到备后端，并记录实际使用的后端。"""

    def __init__(self, primary: Optional[KnowledgeBase], fallback: Optional[KnowledgeBase]):
        if primary is None and fallback is None:
            raise KnowledgeBaseError("没有可用的知识库后端")
        self.primary = primary
        self.fallback = fallback
        self.name = "auto"
        self.last_backend = primary.name if primary else (fallback.name if fallback else "none")
        self.description = (
            (primary.description if primary else "")
            or (fallback.description if fallback else "无可用知识库")
        )

    def search(self, query: str, top_k: int = 4, model: Optional[str] = None) -> List[Passage]:
        errors: List[str] = []
        if self.primary is not None:
            try:
                results = self.primary.search(query, top_k=top_k, model=model)
                self.last_backend = self.primary.name
                return results
            except Exception as exc:
                errors.append(f"{self.primary.name}: {exc}")
        if self.fallback is not None:
            try:
                results = self.fallback.search(query, top_k=top_k, model=model)
                self.last_backend = self.fallback.name
                return results
            except Exception as exc:
                errors.append(f"{self.fallback.name}: {exc}")
        raise KnowledgeBaseError("所有知识库后端均不可用：" + "；".join(errors))

    def fetch(self, chunk_id: str) -> Optional[Passage]:
        for backend in (self.primary, self.fallback):
            if backend is None:
                continue
            try:
                passage = backend.fetch(chunk_id)
                if passage is not None:
                    return passage
            except Exception:
                continue
        return None

    def stats(self) -> Dict[str, Any]:
        return {
            "backend": "auto",
            "last_backend": self.last_backend,
            "primary": self.primary.stats() if self.primary else None,
            "fallback": self.fallback.stats() if self.fallback else None,
            "description": self.description,
        }

    def close(self) -> None:
        for backend in (self.primary, self.fallback):
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def create_knowledge_base(settings: Optional[KnowledgeSettings] = None) -> Optional[KnowledgeBase]:
    """按配置创建知识库；``none`` 或路径缺失时返回 None。"""
    settings = settings or KnowledgeSettings()
    backend = settings.backend

    chroma = ChromaKnowledgeBase(
        persist_path=settings.chroma_path,
        collection_name=settings.collection,
        candidate_limit=settings.candidate_limit,
        max_terms=settings.max_terms,
        timeout=settings.search_timeout,
    )
    jsonl = JsonlKnowledgeBase(settings.jsonl_path)

    chroma_available = Path(settings.chroma_path).expanduser().exists()
    jsonl_available = Path(settings.jsonl_path).expanduser().exists()

    if backend == "none":
        return None
    if backend == "chroma":
        return chroma
    if backend == "jsonl":
        return jsonl
    # auto
    if chroma_available and jsonl_available:
        return AutoKnowledgeBase(chroma, jsonl)
    if chroma_available:
        return chroma
    if jsonl_available:
        return jsonl
    return None


__all__ = [
    "Passage",
    "KnowledgeBase",
    "ChromaKnowledgeBase",
    "JsonlKnowledgeBase",
    "AutoKnowledgeBase",
    "create_knowledge_base",
    "extract_terms",
    "tokenize",
]
