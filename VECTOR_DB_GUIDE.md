# DJI 手册向量数据库构建指南

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      向量数据库系统架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   数据层     │───▶│   嵌入层     │───▶│  向量数据库   │      │
│  │  JSONL文件   │    │  Embedding   │    │   ChromaDB   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                   │                    │              │
│        ▼                   ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ 167万+文本块 │    │ 多模型支持   │    │  余弦相似度   │      │
│  │ 完整元数据   │    │ 批量向量化   │    │  HNSW索引    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 1. 嵌入模型选择

系统支持三种嵌入模型，根据需求和设备选择：

### 1.1 本地轻量模型（推荐）
```bash
# Sentence-Transformers paraphrase-multilingual-MiniLM-L12-v2
set EMBED_MODEL_TYPE=sentence_transformer
set EMBED_MODEL_NAME=./mini_model
set DEVICE=cpu
```
- ✅ **优点**: 免费、隐私安全、无需网络
- ❌ **缺点**: 质量略低于API模型
- 💾 **空间**: ~500MB
- 📊 **维度**: 384维

### 1.2 智谱 AI（中文优化）
```bash
set EMBED_MODEL_TYPE=zhipu
set ZHIPU_API_KEY=your_api_key
set BATCH_SIZE=50
```
- ✅ **优点**: 中文优化效果好
- ❌ **缺点**: 需要API密钥、收费
- 📏 **限制**: ≤50条/批, ≤8192 tokens
- 📊 **维度**: 2048维

### 1.3 OpenAI（多语言）
```bash
set EMBED_MODEL_TYPE=openai
set OPENAI_API_KEY=your_api_key
set EMBED_MODEL_NAME=text-embedding-3-small
set BATCH_SIZE=100
```
- ✅ **优点**: 多语言支持好
- ❌ **缺点**: 需要API密钥、收费
- 📏 **限制**: ≤100条/批, ≤8192 tokens
- 📊 **维度**: 1536维

## 2. 快速开始

### 2.1 使用本地模型（默认）
```bash
# 完整构建流程
python run_vector_db.py build
```

### 2.2 使用API模型
```bash
# OpenAI
set OPENAI_API_KEY=sk-xxx
set EMBED_MODEL_TYPE=openai
python run_vector_db.py build

# 智谱AI
set ZHIPU_API_KEY=xxx
set EMBED_MODEL_TYPE=zhipu
python run_vector_db.py build
```

### 2.3 处理部分数据（测试用）
```bash
# 只处理1000条数据
set MAX_CHUNKS=1000
python run_vector_db.py build
```

## 3. 详细配置

### 3.1 批量向量化配置

```python
EmbeddingConfig(
    model_type="sentence_transformer",  # 模型类型
    model_name="./mini_model",          # 模型路径/名称
    batch_size=32,                       # 批大小
    max_tokens_per_batch=8000,          # 每批最大token数
    device="cpu",                        # 计算设备
    max_retries=3,                       # 失败重试次数
    retry_delay=1.0                      # 重试间隔(秒)
)
```

### 3.2 ChromaDB配置

```python
ChromaConfig(
    persist_path="./chroma_final",       # 存储路径
    collection_name="uav_manual_db",     # 集合名称
    distance_metric="cosine",            # 距离度量
    hnsw_m=16,                          # HNSW参数M
    hnsw_ef_construction=100,           # HNSW构建参数
    hnsw_ef_search=50                   # HNSW搜索参数
)
```

### 3.3 元数据结构

每个向量包含完整元数据：
```json
{
  "doc_id": "文档ID",
  "chunk_id": "块唯一ID",
  "model": "DJI设备型号",
  "title": "文档标题",
  "section": "章节",
  "type": "内容类型",
  "keywords": "关键词",
  "token_count": "token数量",
  "lang": "语言",
  "page": "页码",
  "index_time": "索引时间"
}
```

## 4. 查询接口

### 4.1 基础查询
```python
from vector_db_system import *

# 初始化
model = EmbeddingModelManager(embedding_config)
db = ChromaDBManager(chroma_config)

# 查询
result = db.query(
    query_text="无人机最大飞行高度",
    embedding_model=model,
    n_results=5
)
```

### 4.2 元数据过滤查询
```python
# 按模型过滤
result = db.query(
    query_text="避障功能",
    embedding_model=model,
    n_results=5,
    where_filter={"model": "Mavic 4 Pro"}
)

# 按类型过滤
result = db.query(
    query_text="校准步骤",
    embedding_model=model,
    n_results=5,
    where_filter={"type": "文本", "lang": "zh"}
)
```

### 4.3 查询返回格式
```json
{
  "query": "查询文本",
  "n_results": 5,
  "latency_ms": 15.23,
  "results": [
    {
      "id": "chunk_xxx",
      "rank": 1,
      "similarity": 0.8923,
      "distance": 0.1077,
      "document": "完整文本",
      "document_preview": "预览...",
      "metadata": {...}
    }
  ]
}
```

## 5. 质量验证

### 5.1 验证指标

系统执行6大类验证：

| 验证类型 | 指标 | 说明 |
|---------|------|------|
| 基础统计 | 向量数量、元数据覆盖率 | 验证索引完整性 |
| 自相似性 | Recall@5 | 从索引取文本查询自身 |
| 查询类型 | Precision@5 | 5种查询类型准确率 |
| 边界情况 | 成功率 | 空查询、超长查询等 |
| 元数据过滤 | 准确率 | 过滤条件有效性 |
| 查询延迟 | 平均延迟(ms) | 响应速度 |

### 5.2 运行验证
```bash
# 仅运行验证
python run_vector_db.py verify

# 或查看已有报告
cat retrieval_verification_report.json
```

### 5.3 验证报告解读
```json
{
  "tests": {
    "self_similarity": {
      "recall_at_5": 95.0,      // Recall@5应>90%
      "avg_top1_similarity": 0.95 // 平均相似度应>0.9
    },
    "query_types": {
      "avg_precision_at_5": 85.0, // 平均精确率应>80%
      "avg_latency_ms": 20.0      // 平均延迟应<100ms
    }
  }
}
```

## 6. 产出文件

运行后会生成以下文件：

```
chroma_final/                      # 向量数据库文件
├── chroma.sqlite3                 # 主数据库
└── <uuid>/                        # 向量数据

vector_db_config.json              # 系统配置文档
index_stat.json                    # 索引统计
vectorization_stat.json            # 向量化统计
retrieval_verification_report.json # 检索验证报告
failed_chunks.json                 # 失败记录（如有）
```

## 7. 常见问题

### 7.1 内存不足
```bash
# 减小批大小
set BATCH_SIZE=16

# 限制处理数量
set MAX_CHUNKS=10000
```

### 7.2 查询速度慢
```bash
# 使用GPU（如果可用）
set DEVICE=cuda

# 使用更快的嵌入模型
set EMBED_MODEL_TYPE=sentence_transformer
```

### 7.3 检索结果不相关
- 检查文本块质量
- 尝试不同的嵌入模型
- 调整HNSW参数

### 7.4 API调用失败
```python
# 增加重试次数
EmbeddingConfig(
    max_retries=5,
    retry_delay=2.0
)
```

## 8. 测试查询

```bash
# 运行测试查询
python run_vector_db.py query
```

测试覆盖以下类型：
1. **事实参数查询**: "无人机最大飞行高度"
2. **操作步骤查询**: "如何校准指南针"
3. **故障排查查询**: "遥控器连接失败"
4. **元数据过滤查询**: "Mavic 4 Pro 避障"
5. **模糊语义查询**: "图传卡顿"

## 9. 性能参考

| 模型 | 设备 | 处理速度 | 查询延迟 |
|-----|------|---------|---------|
| MiniLM | CPU | ~100 chunks/s | ~20ms |
| MiniLM | GPU | ~500 chunks/s | ~10ms |
| OpenAI | API | ~50 chunks/s | ~100ms |
| 智谱AI | API | ~30 chunks/s | ~150ms |

---

**提示**: 首次运行建议先用少量数据测试（`set MAX_CHUNKS=1000`），确认无误后再处理全部数据。
