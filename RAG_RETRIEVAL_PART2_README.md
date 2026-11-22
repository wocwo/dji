# RAG检索系统开发 - 第二部分

## 概述

第二部分完成了多路召回、重排序与完整检索流水线的搭建。

## 已完成功能

### 1. MultiPathRetriever 多路检索引擎 (`multi_path_retriever.py`)

**检索通道类型：**
- **SEMANTIC (语义检索)**：主路精准语义检索（向量相似度）
- **KEYWORD (关键词检索)**：实体关键词精准检索（BM25-like）
- **METADATA (元数据检索)**：基于意图的元数据定向过滤检索
- **EXPANDED (扩展查询检索)**：基于同义词扩展的辅路召回

**核心特性：**

#### 动态通道调度 (ChannelScheduler)
- 基于意图自动启停对应通道
- 7种一级意图对应不同的通道权重配置
- 支持二级意图权重调整

```python
# 示例：操作步骤查询的通道权重
{
    RetrievalChannel.SEMANTIC: 1.0,    # 始终启用
    RetrievalChannel.KEYWORD: 0.9,     # 高权重
    RetrievalChannel.METADATA: 0.7,    # 中等权重
    RetrievalChannel.EXPANDED: 0.5     # 低权重
}
```

#### RRF融合算法 (RRFusion)
- **公式**：`RRF_score(d) = Σ(weight_k / (k + rank_d_k))`
- k=60，防止低排名文档分数过高
- 带权重的倒数秩融合，解决不同通道分数维度不一致问题

#### 去重合并
- 同一Chunk在多通道中被召回时，仅保留最高RRF分数
- 合并来源通道信息

### 2. 重排序器 (`reranker.py`)

**支持的模型：**

#### SimpleReranker (轻量级)
- 基于RRF分数和意图加权
- 无需额外模型，CPU运行
- 适用于资源受限场景

#### CrossEncoderReranker (精确级)
- 支持BAAI/bge-reranker系列模型
- 交叉编码器计算查询-文档对相关性
- sigmoid转换分数到0-1范围

**核心功能：**

#### 意图加权
- 根据主意图对匹配类型的Chunk进行分数加权
- 类型匹配 +15%
- 章节匹配 +10%
- 二级意图匹配额外 +10%

#### 阈值过滤
- 默认阈值：0.3
- 过滤掉不相关Chunk，避免噪声进入最终结果

**输出格式：**
```json
{
  "reranked_chunks": [
    {
      "chunk_id": "xxx",
      "final_score": 0.85,
      "rerank_score": 0.80,
      "rrf_score": 0.75,
      "passed_threshold": true
    }
  ],
  "filtered_count": 5,
  "avg_score": 0.72
}
```

### 3. 完整检索流水线 (`rag_pipeline.py`)

**处理流程：**

```
Query → Query Enhancement → Intent Classification → Multi-Path Retrieval → RRF Fusion → Reranking → Final Results
```

**流水线组件：**

| 阶段 | 模块 | 功能 |
|------|------|------|
| 1 | Query Enhancement | 查询清洗、术语标准化、扩展 |
| 2 | Intent Classification | 意图识别、置信度评估 |
| 3 | Multi-Path Retrieval | 4路召回、RRF融合 |
| 4 | Reranking | 精排、意图加权、阈值过滤 |
| 5 | Result Formatting | 格式化输出 |

**性能统计：**
- 各阶段耗时统计
- 召回数量统计
- 过滤数量统计

## 文件清单

```
d:\dji
├── multi_path_retriever.py       # 多路检索引擎
├── reranker.py                    # 重排序器
├── rag_pipeline.py                # 完整检索流水线
├── test_rag_pipeline.py           # 集成测试
└── RAG_RETRIEVAL_PART2_README.md  # 本文档
```

## 使用示例

### 完整检索流程

```python
from rag_pipeline import create_pipeline

# 创建流水线
pipeline = create_pipeline(
    chroma_db_path="./chroma_final",
    embedding_model_type="precomputed",
    embedding_cache_path="embed_cache.npz",
    reranker_type="simple",           # 或 "cross_encoder"
    reranker_threshold=0.3,
    multipath_top_k=50,
    final_top_k=10
)

# 执行检索
result = pipeline.search("Mavic 4 Pro怎么起飞")

# 查看结果
print(pipeline.explain_result(result))
```

### 单独使用多路召回

```python
from multi_path_retriever import MultiPathRetriever
from retrieval_system import IntelligentRetrievalSystem

# 准备查询
system = IntelligentRetrievalSystem()
context = system.process_query("电池怎么充电")

# 多路召回
retriever = MultiPathRetriever(chroma_db, embedding_model)
result = retriever.retrieve(
    query="电池怎么充电",
    enhanced_query=context.enhanced.precise_query,
    expanded_queries=context.enhanced.expanded_queries,
    intent=context.intent
)

print(f"召回{result.total_unique_chunks}个唯一Chunk")
```

### 单独使用重排序

```python
from reranker import create_reranker
from multi_path_retriever import RecalledChunk

# 创建重排序器
reranker = create_reranker(
    model_type="cross_encoder",
    model_name="BAAI/bge-reranker-base",
    threshold=0.3,
    device="cuda"
)

# 重排序
result = reranker.rerank(
    query="电池怎么充电",
    chunks=multipath_result.fused_results,
    intent=intent,
    top_k=10
)

print(f"输出{result.output_count}个文档")
```

## 关键技术细节

### RRF融合参数

```python
RRFusion(k=60)
```

- k=60是经验值，防止低排名文档分数过高
- 权重根据意图动态调整
- 支持4路结果融合

### 意图加权策略

```python
# 基础权重
base_weights = INTENT_CHANNEL_WEIGHTS[intent.primary_intent]

# 二级意图调整
if intent.secondary_intent in SECONDARY_BOOST:
    boost = SECONDARY_BOOST[intent.secondary_intent]
    
# 置信度调整
if intent.confidence < 0.5:
    # 降低元数据通道权重
```

### 阈值过滤

```python
# 默认阈值
threshold = 0.3

# 可配置
reranker = create_reranker(threshold=0.5)  # 更严格
```

## 性能优化

### 1. 通道选择优化
- 根据意图只启用必要的通道
- 低置信度查询减少元数据过滤

### 2. 批处理
- 重排序使用批处理（默认batch_size=8）
- 减少模型推理开销

### 3. 缓存
- 查询增强结果可缓存
- 意图分类结果可缓存

## 测试验证

### 运行集成测试

```bash
python test_rag_pipeline.py
```

### 测试内容

1. **查询增强模块测试**
   - 清洗、标准化、扩展功能
   - 意图识别正确性

2. **多路召回模块测试**
   - 通道调度逻辑
   - RRF融合算法
   - 去重合并

3. **重排序模块测试**
   - 意图加权效果
   - 阈值过滤
   - 分数排序

4. **流水线集成测试**
   - 各模块协同工作
   - 端到端流程

## 与第一部分的集成

```
第一部分 (已完成)
├── query_enhancement.py      ← 复用
├── intent_classifier.py      ← 复用
└── filter_generator.py       ← 复用

第二部分 (新增)
├── multi_path_retriever.py   ← 新增
├── reranker.py               ← 新增
└── rag_pipeline.py           ← 整合
```

## 后续优化方向

1. **关键词检索优化**
   - 集成jieba分词
   - 构建倒排索引
   - 优化BM25算法

2. **重排序模型优化**
   - 支持更多重排序模型（如cohere-rerank）
   - 模型蒸馏减小体积
   - ONNX加速推理

3. **混合检索策略**
   - 结合稀疏检索（BM25）和稠密检索（向量）
   - 动态调整融合权重

4. **结果多样性**
   - MMR（最大边缘相关性）算法
   - 避免结果过于相似

5. **缓存机制**
   - 查询结果缓存
   - 嵌入向量缓存

## 依赖

```python
# 必需
import numpy as np
from sentence_transformers import CrossEncoder  # 可选，用于CrossEncoderReranker

# 内部模块
from retrieval_system import IntelligentRetrievalSystem
from intent_classifier import IntentClassification
from multi_path_retriever import MultiPathRetriever
from reranker import create_reranker
```

## 版本历史

- **v2.0.0** (2026-03-30): 完成第二部分 - 多路召回、重排序、完整流水线
- **v1.1.0** (2026-03-30): 优化缩写查询识别率至100%
- **v1.0.0** (2026-03-30): 完成第一部分 - 查询增强、意图分类、过滤生成
