"""
端到端真实检索 + 上下文组装 + Prompt 生成测试

运行要求：
- chroma_final/ 向量库可正常访问
- embed_cache.npz 嵌入缓存文件存在
- 各模块导入无异常

测试内容：
1. 初始化完整 RAG 流水线
2. 对多个典型查询执行检索
3. 输出生成的 Markdown 上下文和最终 Prompt
"""

import os
import sys


def check_environment():
    """检查运行环境"""
    required_paths = {
        "chroma_final": "./chroma_final/chroma.sqlite3",
        "embed_cache": "./embed_cache.npz"
    }
    
    all_ok = True
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    for name, path in required_paths.items():
        exists = os.path.exists(path)
        status = "[OK] 存在" if exists else "[MISSING] 缺失"
        print(f"  {name}: {path} {status}")
        if not exists:
            all_ok = False
    
    print()
    return all_ok


def test_pipeline_with_prompts():
    """测试完整流水线并打印生成的 Prompt"""
    from rag_pipeline import create_pipeline
    
    print("=" * 60)
    print("初始化 RAG 流水线...")
    print("=" * 60)
    
    # 初始化流水线（使用预计算缓存，无需加载大模型）
    pipeline = create_pipeline(
        chroma_db_path="./chroma_final",
        embedding_model_type="precomputed",
        embedding_cache_path="embed_cache.npz",
        reranker_type="simple",      # 轻量级重排序，无需 GPU
        reranker_threshold=0.3,
        multipath_top_k=20,
        final_top_k=5
    )
    
    print("初始化完成！\n")
    
    # 典型测试查询（覆盖不同意图）
    test_queries = [
        ("Mavic 4 Pro 最大飞行高度是多少？", "参数查询"),
        ("怎么进行指南针校准？", "操作步骤查询"),
        ("电池充不满怎么办？", "故障排查查询"),
        ("禁飞区有哪些规定？", "安全合规查询"),
    ]
    
    for query, expected_intent in test_queries:
        print("\n" + "=" * 70)
        print(f"【查询】{query}")
        print(f"【预期意图】{expected_intent}")
        print("=" * 70)
        
        try:
            result = pipeline.search(query)
            
            # 1. 打印检索统计
            print(f"\n>> 意图识别: {result.context.intent.primary_intent.value} -> {result.context.intent.secondary_intent.value}")
            print(f">> 召回 Chunk 数: {result.multipath_result.total_unique_chunks}")
            print(f">> 最终通过阈值: {result.rerank_result.output_count} / {result.rerank_result.input_count}")
            print(f">> 总耗时: {result.total_time_ms:.2f}ms")
            
            if result.generation_context:
                gen = result.generation_context
                print(f"\n>> 上下文组装:")
                print(f"   - 组装 Chunk 数: {gen.assembled_context.chunk_count}")
                print(f"   - Markdown 长度: {gen.assembled_context.total_chars} 字符")
                print(f"   - Prompt 总长度: {gen.prompt_metadata.get('prompt_chars', 0)} 字符")
                print(f"   - 组装耗时: {gen.assembled_context.assembly_time_ms:.2f}ms")
                
                # 2. 打印生成的 Markdown 上下文（截取前 800 字符）
                print(f"\n{'─' * 60}")
                print("【生成的 Markdown 上下文（前 800 字符）】")
                print("─" * 60)
                print(gen.assembled_context.markdown_text[:800])
                if len(gen.assembled_context.markdown_text) > 800:
                    print("\n...[截断]")
                
                # 3. 打印最终 Prompt（截取前 1200 字符）
                print(f"\n{'─' * 60}")
                print("【生成的 Prompt（前 1200 字符）】")
                print("─" * 60)
                print(gen.prompt[:1200])
                if len(gen.prompt) > 1200:
                    print("\n...[截断]")
            else:
                print("\n[WARNING] 未生成 generation_context")
                
        except Exception as e:
            print(f"\n[ERROR] 查询失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    if not check_environment():
        print("环境检查未通过，请确认 chroma_final/ 和 embed_cache.npz 存在后再运行。")
        sys.exit(1)
    
    test_pipeline_with_prompts()
