"""
DJI 手册向量数据库构建 - 简化运行脚本
支持从环境变量读取API密钥
"""

import os
import sys
import json
from vector_db_system import (
    VectorDBConfig, EmbeddingConfig, ChromaConfig,
    EmbeddingModelManager, BatchEmbeddingProcessor,
    ChromaDBManager, QualityValidator, load_chunks
)


def get_env_config():
    """从环境变量获取配置"""
    model_type = os.getenv("EMBED_MODEL_TYPE", "sentence_transformer")
    
    config = {
        "model_type": model_type,
        "model_name": os.getenv("EMBED_MODEL_NAME", "./mini_model"),
        "batch_size": int(os.getenv("BATCH_SIZE", "32")),
        "device": os.getenv("DEVICE", "cpu"),
        "max_chunks": os.getenv("MAX_CHUNKS"),
    }
    
    # API密钥
    if model_type == "openai":
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        config["api_base"] = os.getenv("OPENAI_API_BASE")
        config["model_name"] = os.getenv("EMBED_MODEL_NAME", "text-embedding-3-small")
    elif model_type == "zhipu":
        config["api_key"] = os.getenv("ZHIPU_API_KEY")
        config["api_base"] = os.getenv("ZHIPU_API_BASE")
        config["model_name"] = os.getenv("EMBED_MODEL_NAME", "embedding-3")
    
    return config


def print_usage():
    """打印使用说明"""
    print("""
DJI 向量数据库构建工具
======================

使用方法:
    python run_vector_db.py [命令] [选项]

命令:
    build    构建向量数据库（完整流程）
    query    测试查询
    verify   仅运行验证

选项（环境变量）:
    EMBED_MODEL_TYPE     嵌入模型类型: sentence_transformer / openai / zhipu (默认: sentence_transformer)
    EMBED_MODEL_NAME     模型名称或路径 (默认: ./mini_model)
    BATCH_SIZE           批处理大小 (默认: 32)
    DEVICE               计算设备: cpu / cuda (默认: cpu)
    MAX_CHUNKS           最大处理块数 (默认: None, 处理全部)
    
    # OpenAI 配置
    OPENAI_API_KEY       OpenAI API密钥
    OPENAI_API_BASE      OpenAI API基础URL（可选）
    
    # 智谱AI 配置
    ZHIPU_API_KEY        智谱AI API密钥
    ZHIPU_API_BASE       智谱AI API基础URL（可选）

示例:
    # 使用本地模型构建
    python run_vector_db.py build
    
    # 使用OpenAI API构建（设置环境变量）
    set OPENAI_API_KEY=your_key
    set EMBED_MODEL_TYPE=openai
    python run_vector_db.py build
    
    # 使用智谱AI构建
    set ZHIPU_API_KEY=your_key
    set EMBED_MODEL_TYPE=zhipu
    set BATCH_SIZE=50
    python run_vector_db.py build
    
    # 只处理10000条数据
    set MAX_CHUNKS=10000
    python run_vector_db.py build
    """)


def cmd_build(env_config):
    """构建向量数据库"""
    print("="*60)
    print("DJI 手册向量数据库构建")
    print("="*60)
    
    # 创建配置
    embed_config = EmbeddingConfig(
        model_type=env_config["model_type"],
        model_name=env_config["model_name"],
        api_key=env_config.get("api_key"),
        api_base=env_config.get("api_base"),
        batch_size=env_config["batch_size"],
        device=env_config["device"]
    )
    
    config = VectorDBConfig(
        embedding=embed_config,
        max_chunks=int(env_config["max_chunks"]) if env_config["max_chunks"] else None
    )
    
    # 1. 加载数据
    chunks = load_chunks("dji_chunks_final.jsonl", config.max_chunks)
    
    # 2. 初始化模型
    print("\n" + "-"*60)
    model_manager = EmbeddingModelManager(config.embedding)
    
    # 3. 向量化
    print("\n" + "-"*60)
    processor = BatchEmbeddingProcessor(model_manager, config.embedding)
    embed_result = processor.process_chunks(chunks)
    
    # 保存统计
    with open("vectorization_stat.json", "w", encoding="utf-8") as f:
        json.dump(embed_result["stats"], f, ensure_ascii=False, indent=2)
    
    if embed_result["failed_chunks"]:
        with open("failed_chunks.json", "w", encoding="utf-8") as f:
            json.dump(embed_result["failed_chunks"], f, ensure_ascii=False, indent=2)
    
    # 4. 构建索引
    print("\n" + "-"*60)
    db_manager = ChromaDBManager(config.chroma)
    db_manager.add_embeddings(embed_result)
    
    # 5. 验证
    print("\n" + "-"*60)
    validator = QualityValidator(db_manager, model_manager)
    validator.run_full_validation(chunks)
    
    print("\n" + "="*60)
    print("✅ 构建完成！")
    print("="*60)


def cmd_query():
    """测试查询"""
    from vector_db_system import EmbeddingConfig, ChromaConfig
    
    env_config = get_env_config()
    
    # 加载模型和数据库
    embed_config = EmbeddingConfig(
        model_type=env_config["model_type"],
        model_name=env_config["model_name"],
        api_key=env_config.get("api_key"),
        api_base=env_config.get("api_base"),
        device=env_config["device"]
    )
    
    model_manager = EmbeddingModelManager(embed_config)
    db_manager = ChromaDBManager(ChromaConfig())
    
    # 测试查询
    test_queries = [
        "无人机最大飞行高度",
        "如何校准指南针",
        "遥控器连接失败",
        "Mavic 4 Pro 避障",
        "图传卡顿"
    ]
    
    print("\n" + "="*60)
    print("查询测试")
    print("="*60)
    
    for query in test_queries:
        print(f"\n查询: {query}")
        result = db_manager.query(query, model_manager, n_results=3)
        print(f"  延迟: {result['latency_ms']}ms")
        print(f"  结果数: {len(result['results'])}")
        for r in result['results'][:2]:
            print(f"  [{r['rank']}] 相似度: {r['similarity']} | {r['document_preview'][:80]}...")


def cmd_verify():
    """仅运行验证"""
    from vector_db_system import EmbeddingConfig, ChromaConfig
    
    env_config = get_env_config()
    
    embed_config = EmbeddingConfig(
        model_type=env_config["model_type"],
        model_name=env_config["model_name"],
        api_key=env_config.get("api_key"),
        api_base=env_config.get("api_base"),
        device=env_config["device"]
    )
    
    model_manager = EmbeddingModelManager(embed_config)
    db_manager = ChromaDBManager(ChromaConfig())
    
    # 加载部分数据进行验证
    chunks = load_chunks("dji_chunks_final.jsonl", max_chunks=100)
    
    validator = QualityValidator(db_manager, model_manager)
    validator.run_full_validation(chunks)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help", "help"]:
        print_usage()
        return
    
    command = sys.argv[1]
    env_config = get_env_config()
    
    if command == "build":
        cmd_build(env_config)
    elif command == "query":
        cmd_query()
    elif command == "verify":
        cmd_verify()
    else:
        print(f"未知命令: {command}")
        print_usage()


if __name__ == "__main__":
    main()
