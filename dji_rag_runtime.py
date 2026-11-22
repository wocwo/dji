"""
DJI RAG 运行时增强组件

包含三类能力：
1. DJIRAGEvaluator: 单条/批量答案评估（多维指标）
2. DJIRAGCacheManager: 内存 + 本地磁盘两级缓存管理
3. DJIRAGBatchProcessor: 高效容错批量处理引擎（线程/进程）
"""

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from cachetools import TTLCache
from diskcache import Cache


logger = logging.getLogger(__name__)


# =============================================================================
# 评估能力
# =============================================================================


@dataclass
class ReadabilityMetrics:
    """可读性指标"""

    avg_sentence_length: float
    avg_word_length: float
    sentence_count: int
    word_count: int
    char_count: int


@dataclass
class EvaluationScore:
    """单条评估结果"""

    query: str
    answer: str
    reference: str = ""
    rouge_precision: float = 0.0
    rouge_recall: float = 0.0
    rouge_f1: float = 0.0
    key_info_coverage: float = 0.0
    hallucination_score: float = 1.0
    readability: Optional[ReadabilityMetrics] = None
    overall_score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BatchEvaluationReport:
    """批量评估报告"""

    total: int
    avg_overall_score: float
    avg_rouge_f1: float
    avg_key_info_coverage: float
    avg_hallucination_score: float
    avg_sentence_length: float
    passed_count: int
    fail_count: int
    results: List[EvaluationScore] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "avg_overall_score": self.avg_overall_score,
            "avg_rouge_f1": self.avg_rouge_f1,
            "avg_key_info_coverage": self.avg_key_info_coverage,
            "avg_hallucination_score": self.avg_hallucination_score,
            "avg_sentence_length": self.avg_sentence_length,
            "passed_count": self.passed_count,
            "fail_count": self.fail_count,
            "results": [asdict(item) for item in self.results],
            "generated_at": self.generated_at,
        }


class DJIRAGEvaluator:
    """
    答案评估器（支持单条与批量）

    评估维度：
    - 文本相似性（ROUGE 风格 token overlap）
    - 关键信息覆盖率
    - 幻觉检测（answer 是否包含参考中不存在的大量信息）
    - 可读性（平均句长、平均词长等）
    """

    def __init__(
        self,
        key_info_extractor: Optional[Callable[[str], Sequence[str]]] = None,
        pass_threshold: float = 0.65,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.key_info_extractor = key_info_extractor or self._default_key_info_extractor
        self.pass_threshold = pass_threshold
        self.weights = weights or {
            "rouge_f1": 0.35,
            "key_info_coverage": 0.30,
            "hallucination_score": 0.25,
            "readability_score": 0.10,
        }

    def evaluate(
        self,
        query: str,
        answer: str,
        reference: str,
        key_facts: Optional[Sequence[str]] = None,
    ) -> EvaluationScore:
        """单条评估"""
        clean_answer = answer.strip()
        clean_ref = reference.strip()

        precision, recall, rouge_f1 = self._compute_rouge_like(clean_answer, clean_ref)

        facts = list(key_facts) if key_facts is not None else list(self.key_info_extractor(clean_ref))
        key_info_coverage = self._compute_key_info_coverage(clean_answer, facts)
        hallucination_score, hallucination_hits = self._compute_hallucination_score(clean_answer, clean_ref)
        readability = self._compute_readability(clean_answer)
        readability_score = self._normalize_readability(readability.avg_sentence_length)

        overall = (
            rouge_f1 * self.weights["rouge_f1"]
            + key_info_coverage * self.weights["key_info_coverage"]
            + hallucination_score * self.weights["hallucination_score"]
            + readability_score * self.weights["readability_score"]
        )

        return EvaluationScore(
            query=query,
            answer=answer,
            reference=reference,
            rouge_precision=round(precision, 4),
            rouge_recall=round(recall, 4),
            rouge_f1=round(rouge_f1, 4),
            key_info_coverage=round(key_info_coverage, 4),
            hallucination_score=round(hallucination_score, 4),
            readability=readability,
            overall_score=round(overall, 4),
            details={
                "key_facts": facts,
                "hallucination_candidates": hallucination_hits,
                "pass": overall >= self.pass_threshold,
            },
        )

    def evaluate_batch(
        self,
        items: Sequence[Dict[str, Any]],
    ) -> BatchEvaluationReport:
        """
        批量评估

        items 元素格式:
        {
          "query": "...",
          "answer": "...",
          "reference": "...",
          "key_facts": ["...", ...]   # 可选
        }
        """
        results: List[EvaluationScore] = []
        for item in items:
            result = self.evaluate(
                query=item.get("query", ""),
                answer=item.get("answer", ""),
                reference=item.get("reference", ""),
                key_facts=item.get("key_facts"),
            )
            results.append(result)

        if not results:
            return BatchEvaluationReport(
                total=0,
                avg_overall_score=0.0,
                avg_rouge_f1=0.0,
                avg_key_info_coverage=0.0,
                avg_hallucination_score=0.0,
                avg_sentence_length=0.0,
                passed_count=0,
                fail_count=0,
                results=[],
            )

        avg_overall = mean([x.overall_score for x in results])
        avg_rouge = mean([x.rouge_f1 for x in results])
        avg_key = mean([x.key_info_coverage for x in results])
        avg_hal = mean([x.hallucination_score for x in results])
        avg_sent_len = mean([x.readability.avg_sentence_length for x in results if x.readability])
        passed = sum(1 for x in results if x.details.get("pass"))

        return BatchEvaluationReport(
            total=len(results),
            avg_overall_score=round(avg_overall, 4),
            avg_rouge_f1=round(avg_rouge, 4),
            avg_key_info_coverage=round(avg_key, 4),
            avg_hallucination_score=round(avg_hal, 4),
            avg_sentence_length=round(avg_sent_len, 4),
            passed_count=passed,
            fail_count=len(results) - passed,
            results=results,
        )

    def _compute_rouge_like(self, answer: str, reference: str) -> Tuple[float, float, float]:
        answer_tokens = self._tokenize(answer)
        ref_tokens = self._tokenize(reference)
        if not answer_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
        overlap = sum((answer_tokens & ref_tokens).values())
        precision = overlap / sum(answer_tokens.values())
        recall = overlap / sum(ref_tokens.values())
        if precision + recall == 0:
            return precision, recall, 0.0
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def _compute_key_info_coverage(self, answer: str, key_facts: Sequence[str]) -> float:
        if not key_facts:
            return 0.0
        answer_lower = answer.lower()
        covered = sum(1 for fact in key_facts if fact and fact.lower() in answer_lower)
        return covered / len(key_facts)

    def _compute_hallucination_score(self, answer: str, reference: str) -> Tuple[float, List[str]]:
        """简单幻觉检测：答案中独有 token 占比越高，分数越低"""
        answer_tokens = set(self._tokenize(answer).keys())
        ref_tokens = set(self._tokenize(reference).keys())
        if not answer_tokens:
            return 0.0, []
        extras = sorted([t for t in answer_tokens if t not in ref_tokens and len(t) > 1])
        extra_ratio = len(extras) / len(answer_tokens)
        score = max(0.0, 1.0 - extra_ratio)
        return score, extras[:30]

    def _compute_readability(self, text: str) -> ReadabilityMetrics:
        sentences = [s.strip() for s in re.split(r"[。！？!?\.]+", text) if s.strip()]
        words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text)
        sentence_count = max(len(sentences), 1)
        word_count = len(words)
        char_count = len(text)
        avg_sentence_length = word_count / sentence_count if sentence_count else 0.0
        avg_word_length = (
            sum(len(w) for w in words if isinstance(w, str)) / word_count if word_count else 0.0
        )
        return ReadabilityMetrics(
            avg_sentence_length=round(avg_sentence_length, 4),
            avg_word_length=round(avg_word_length, 4),
            sentence_count=sentence_count,
            word_count=word_count,
            char_count=char_count,
        )

    def _normalize_readability(self, avg_sentence_length: float) -> float:
        """
        句长适中（8~25）得分最高，过短或过长下降。
        """
        if 8 <= avg_sentence_length <= 25:
            return 1.0
        if avg_sentence_length < 8:
            return max(0.0, avg_sentence_length / 8.0)
        return max(0.0, 1.0 - ((avg_sentence_length - 25) / 25.0))

    @staticmethod
    def _tokenize(text: str):
        from collections import Counter

        tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", text.lower())
        return Counter(tokens)

    @staticmethod
    def _default_key_info_extractor(text: str) -> Sequence[str]:
        """
        默认关键事实抽取：
        - 基于句号分句
        - 取较长且包含数字/动词指令特征的句子
        """
        candidates = [s.strip() for s in re.split(r"[。！？!?]", text) if s.strip()]
        selected: List[str] = []
        for sentence in candidates:
            if len(sentence) >= 8 and (re.search(r"\d", sentence) or re.search(r"(需要|必须|请|步骤|检查)", sentence)):
                selected.append(sentence)
        return selected[:8]


# =============================================================================
# 缓存管理能力
# =============================================================================


class DJIRAGCacheManager:
    """
    两级缓存管理器

    L1: 内存缓存（cachetools TTLCache，TTL + LRU）
    L2: 本地磁盘缓存（diskcache，TTL + 容量淘汰）
    """

    def __init__(
        self,
        memory_maxsize: int = 1000,
        memory_ttl_seconds: int = 3600,
        disk_path: str = ".cache/dji_rag",
        disk_size_limit_gb: int = 10,
        disk_ttl_seconds: int = 24 * 3600,
        version: str = "v1",
    ):
        self.memory_cache = TTLCache(maxsize=memory_maxsize, ttl=memory_ttl_seconds)
        os.makedirs(disk_path, exist_ok=True)
        self.disk_cache = Cache(directory=disk_path, size_limit=disk_size_limit_gb * 1024 * 1024 * 1024)
        self.disk_ttl_seconds = disk_ttl_seconds
        self.version = version
        self.stats = {
            "memory_hit": 0,
            "disk_hit": 0,
            "miss": 0,
            "set_count": 0,
            "last_cleanup_at": None,
        }

    def build_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        raw = json.dumps(
            {
                "query": query,
                "context": context or {},
                "version": self.version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        start = time.time()
        if key in self.memory_cache:
            self.stats["memory_hit"] += 1
            value = self.memory_cache[key]
            logger.debug("[Cache] memory hit key=%s cost=%.2fms", key[:10], (time.time() - start) * 1000)
            return value

        value = self.disk_cache.get(key, default=None)
        if value is not None:
            self.stats["disk_hit"] += 1
            # 回填内存缓存，提高高频请求速度
            self.memory_cache[key] = value
            logger.debug("[Cache] disk hit key=%s cost=%.2fms", key[:10], (time.time() - start) * 1000)
            return value

        self.stats["miss"] += 1
        logger.debug("[Cache] miss key=%s cost=%.2fms", key[:10], (time.time() - start) * 1000)
        return None

    def set(self, key: str, value: Any):
        self.memory_cache[key] = value
        self.disk_cache.set(key, value, expire=self.disk_ttl_seconds)
        self.stats["set_count"] += 1

    def get_or_set(self, key: str, factory: Callable[[], Any]) -> Any:
        value = self.get(key)
        if value is not None:
            return value
        value = factory()
        self.set(key, value)
        return value

    def clear(self, memory_only: bool = False):
        self.memory_cache.clear()
        if not memory_only:
            self.disk_cache.clear()
        self.stats["last_cleanup_at"] = datetime.now().isoformat()

    def cleanup_expired(self):
        # diskcache.expire 会删除过期项并返回删除数量
        removed = self.disk_cache.expire()
        self.stats["last_cleanup_at"] = datetime.now().isoformat()
        return removed

    def set_version(self, version: str):
        """切换缓存版本（key会自动隔离）"""
        self.version = version

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["memory_hit"] + self.stats["disk_hit"] + self.stats["miss"]
        hit_rate = (self.stats["memory_hit"] + self.stats["disk_hit"]) / total if total else 0.0
        return {
            **self.stats,
            "memory_size": len(self.memory_cache),
            "disk_volume": self.disk_cache.volume(),
            "version": self.version,
            "hit_rate": round(hit_rate, 4),
        }

    def close(self):
        self.disk_cache.close()


# =============================================================================
# 批处理能力（可选）
# =============================================================================


def _execute_with_retry(
    item: Any,
    handler: Callable[[Any], Any],
    retries: int,
    retry_delay_seconds: float,
    fallback: Optional[Callable[[Any, Exception], Any]],
) -> Dict[str, Any]:
    attempt = 0
    last_error: Optional[Exception] = None
    while attempt <= retries:
        try:
            result = handler(item)
            return {
                "ok": True,
                "item": item,
                "result": result,
                "attempts": attempt + 1,
                "error": None,
            }
        except Exception as exc:  # pylint: disable=broad-except
            last_error = exc
            attempt += 1
            if attempt <= retries:
                time.sleep(retry_delay_seconds)

    degraded = None
    if fallback and last_error is not None:
        try:
            degraded = fallback(item, last_error)
            return {
                "ok": True,
                "item": item,
                "result": degraded,
                "attempts": retries + 1,
                "error": f"degraded_from_error: {last_error}",
            }
        except Exception as fallback_exc:  # pylint: disable=broad-except
            last_error = fallback_exc

    return {
        "ok": False,
        "item": item,
        "result": None,
        "attempts": retries + 1,
        "error": str(last_error) if last_error else "unknown_error",
    }


class DJIRAGBatchProcessor:
    """
    高效容错批量处理器

    特性：
    - 自动并发：线程=CPU*2，进程=CPU-1
    - 失败重试 + 降级
    - 失败日志记录并继续处理其余任务
    """

    def __init__(
        self,
        mode: str = "thread",
        max_workers: Optional[int] = None,
        retries: int = 2,
        retry_delay_seconds: float = 0.1,
        failure_log_path: str = "batch_failures.log",
    ):
        if mode not in {"thread", "process"}:
            raise ValueError("mode must be 'thread' or 'process'")
        self.mode = mode
        cpu = os.cpu_count() or 2
        if max_workers is None:
            self.max_workers = max(1, cpu * 2) if mode == "thread" else max(1, cpu - 1)
        else:
            self.max_workers = max(1, int(max_workers))
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.failure_log_path = failure_log_path

    def process(
        self,
        items: Iterable[Any],
        handler: Callable[[Any], Any],
        fallback: Optional[Callable[[Any, Exception], Any]] = None,
    ) -> Dict[str, Any]:
        item_list = list(items)
        if not item_list:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "results": [],
            }

        worker = partial(
            _execute_with_retry,
            handler=handler,
            retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            fallback=fallback,
        )

        executor_cls = ThreadPoolExecutor if self.mode == "thread" else ProcessPoolExecutor
        results: List[Dict[str, Any]] = []
        failed_items: List[Dict[str, Any]] = []

        start = time.time()
        with executor_cls(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(worker, item): item for item in item_list}
            for fut in as_completed(future_map):
                out = fut.result()
                results.append(out)
                if not out["ok"]:
                    failed_items.append(out)

        elapsed_ms = (time.time() - start) * 1000

        if failed_items:
            self._append_failures(failed_items)

        return {
            "mode": self.mode,
            "max_workers": self.max_workers,
            "total": len(item_list),
            "success": len(item_list) - len(failed_items),
            "failed": len(failed_items),
            "elapsed_ms": round(elapsed_ms, 2),
            "results": results,
        }

    def _append_failures(self, failed_items: List[Dict[str, Any]]):
        with open(self.failure_log_path, "a", encoding="utf-8") as f:
            for item in failed_items:
                f.write(
                    json.dumps(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "item": item.get("item"),
                            "error": item.get("error"),
                            "attempts": item.get("attempts"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
