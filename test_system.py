"""测试向量数据库系统"""
import sys
sys.path.insert(0, '.')
from vector_db_system import *

# 测试配置
config = VectorDBConfig(
    embedding=EmbeddingConfig(
        model_type='sentence_transformer',
        model_name='./mini_model',
        batch_size=32,
        device='cpu'
    ),
    max_chunks=100
)

print('测试1: 加载数据')
chunks = load_chunks('dji_chunks_final.jsonl', 100)
print(f'成功加载 {len(chunks)} 条数据')

print('\n测试2: 初始化模型')
model = EmbeddingModelManager(config.embedding)

print('\n测试3: 向量化 (10条样本)')
processor = BatchEmbeddingProcessor(model, config.embedding)
result = processor.process_chunks(chunks[:10])
print(f"向量化完成: {result['stats']}")

print('\n测试4: 构建索引')
db = ChromaDBManager(config.chroma)
db.add_embeddings(result)

print('\n测试5: 查询测试')
query_result = db.query('无人机飞行高度', model, n_results=3)
print(f"查询结果: {len(query_result['results'])} 条")
for r in query_result['results'][:2]:
    print(f"  [{r['rank']}] 相似度: {r['similarity']}")

print('\n测试6: 质量验证')
validator = QualityValidator(db, model)
report = validator.run_full_validation(chunks[:20])

print('\n✅ 所有测试通过！')
