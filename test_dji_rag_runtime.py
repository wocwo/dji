import time

from dji_rag_runtime import DJIRAGBatchProcessor, DJIRAGCacheManager, DJIRAGEvaluator


def test_evaluator_single_and_batch():
    evaluator = DJIRAGEvaluator()
    single = evaluator.evaluate(
        query="Mavic 4 Pro 怎么起飞？",
        answer="起飞前需要检查电量和GPS信号，然后启动电机起飞。",
        reference="起飞前必须检查电量和GPS信号，确认安全后启动电机。",
    )
    assert 0.0 <= single.rouge_f1 <= 1.0
    assert 0.0 <= single.key_info_coverage <= 1.0
    assert 0.0 <= single.hallucination_score <= 1.0
    assert single.readability is not None

    batch = evaluator.evaluate_batch(
        [
            {
                "query": "Q1",
                "answer": "先检查电池，再起飞。",
                "reference": "起飞前需要检查电池。",
            },
            {
                "query": "Q2",
                "answer": "请检查遥控器连接状态。",
                "reference": "飞行前需检查遥控器连接状态。",
            },
        ]
    )
    assert batch.total == 2
    assert len(batch.results) == 2


def test_cache_manager_two_levels():
    manager = DJIRAGCacheManager(
        memory_maxsize=2,
        memory_ttl_seconds=3600,
        disk_path=".cache/test_dji_rag_runtime",
        disk_size_limit_gb=1,
        disk_ttl_seconds=3600,
    )
    key = manager.build_key("如何返航", {"model": "Mavic 4 Pro"})
    payload = {"answer": "长按返航键"}

    manager.set(key, payload)
    assert manager.get(key) == payload

    manager.memory_cache.clear()
    assert manager.get(key) == payload
    stats = manager.get_stats()
    assert stats["disk_hit"] >= 1
    manager.clear(memory_only=False)
    manager.close()


def _handler(item):
    if item == "bad":
        raise ValueError("boom")
    return f"ok:{item}"


def _fallback(item, _err):
    return f"fallback:{item}"


def test_batch_processor_retry_and_fallback():
    processor = DJIRAGBatchProcessor(mode="thread", retries=1, retry_delay_seconds=0.01)
    result = processor.process(["a", "bad", "b"], handler=_handler, fallback=_fallback)
    assert result["total"] == 3
    assert result["success"] == 3
    outputs = [x["result"] for x in result["results"]]
    assert any(str(out).startswith("fallback:bad") for out in outputs)


if __name__ == "__main__":
    start = time.time()
    test_evaluator_single_and_batch()
    test_cache_manager_two_levels()
    test_batch_processor_retry_and_fallback()
    print(f"[OK] test_dji_rag_runtime passed in {(time.time() - start):.2f}s")
