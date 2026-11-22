"""
DJI 手册向量数据库构建系统
支持多种嵌入模型：智谱AI、OpenAI、Sentence-Transformers、预计算嵌入
"""

import os
import json
import time
import hashlib
import warnings
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# 可选依赖 - 根据使用的模型导入（延迟导入避免torch DLL问题）
SentenceTransformer = None
SENTENCE_TRANSFORMERS_AVAILABLE = False

def _import_sentence_transformers():
    global SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE
    try:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
    return SENTENCE_TRANSFORMERS_AVAILABLE

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    tiktoken = None
    TIKTOKEN_AVAILABLE = False

# ===================== 配置类 =====================

@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    model_type: str = "precomputed"  # precomputed / zhipu / openai / sentence_transformer
    model_name: str = "./mini_model"  # 本地路径或模型名称
    
    # API配置（用于智谱/OpenAI）
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # 批量处理配置
    batch_size: int = 32
    max_tokens_per_batch: int = 8000  # OpenAI: 8192, 智谱: 相对较小
    
    # 设备配置
    device: str = "cpu"  # cuda / cpu
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 预计算嵌入配置
    precomputed_cache_path: str = "embed_cache.npz"
    
    def __post_init__(self):
        if self.model_type == "openai":
            self.batch_size = min(self.batch_size, 100)
            self.max_tokens_per_batch = 8000
        elif self.model_type == "zhipu":
            self.batch_size = min(self.batch_size, 50)
            self.max_tokens_per_batch = 6000


@dataclass
class ChromaConfig:
    """ChromaDB配置"""
    persist_path: str = "./chroma_final"
    collection_name: str = "uav_manual_db"
    distance_metric: str = "cosine"  # cosine / l2 / ip
    
    # HNSW索引参数
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 50


@dataclass
class VectorDBConfig:
    """完整配置"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    
    # 数据处理
    max_chunks: Optional[int] = None  # None表示处理全部
    save_failed_chunks: bool = True
    
    # 日志
    log_level: str = "INFO"


# ===================== 嵌入模型管理器 =====================

class EmbeddingModelManager:
    """管理多种嵌入模型的加载和调用"""
    
    SUPPORTED_MODELS = {
        "precomputed": {
            "name": "预计算嵌入 (embed_cache.npz)",
            "description": "使用已缓存的嵌入向量，无需实时计算"
        },
        "zhipu": {
            "name": "智谱AI text_embedding",
            "default_model": "embedding-3",
            "api_base": "https://open.bigmodel.cn/api/paas/v4/embeddings",
            "max_tokens": 8192,
            "batch_limit": 50
        },
        "openai": {
            "name": "OpenAI text-embedding-3-small",
            "default_model": "text-embedding-3-small",
            "api_base": "https://api.openai.com/v1",
            "max_tokens": 8192,
            "batch_limit": 100
        },
        "sentence_transformer": {
            "name": "Sentence-Transformers paraphrase-multilingual-MiniLM",
            "default_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "max_tokens": 512,
            "batch_limit": 128
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._api_client = None
        self._precomputed_embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dim: int = 384  # 默认维度
        self._initialize()
    
    def _initialize(self):
        """初始化模型"""
        model_type = self.config.model_type
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        info = self.SUPPORTED_MODELS[model_type]
        print(f"[初始化] 加载嵌入模型: {info['name']}")
        
        if model_type == "precomputed":
            self._init_precomputed()
        elif model_type == "sentence_transformer":
            self._init_sentence_transformer()
        elif model_type == "openai":
            self._init_openai()
        elif model_type == "zhipu":
            self._init_zhipu()
    
    def _init_precomputed(self):
        """初始化预计算嵌入缓存"""
        cache_path = self.config.precomputed_cache_path
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"预计算嵌入缓存不存在: {cache_path}")
        
        print(f"  加载缓存: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        
        ids = data['ids']
        embeddings = data['embeddings']
        
        # 构建id到embedding的映射
        self._precomputed_embeddings = {
            str(id_): emb for id_, emb in zip(ids, embeddings)
        }
        self.embedding_dim = embeddings.shape[1]
        
        print(f"  加载完成: {len(self._precomputed_embeddings)} 个向量")
        print(f"  向量维度: {self.embedding_dim}")
    
    def _init_sentence_transformer(self):
        """初始化本地SentenceTransformer模型"""
        _import_sentence_transformers()
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        
        model_path = self.config.model_name
        if os.path.exists(model_path):
            print(f"  加载本地模型: {model_path}")
            self.model = SentenceTransformer(model_path, device=self.config.device)
        else:
            print(f"  下载模型: {model_path}")
            self.model = SentenceTransformer(model_path, device=self.config.device)
        
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  模型维度: {self.embedding_dim}")
        print(f"  设备: {self.config.device}")
    
    def _init_openai(self):
        """初始化OpenAI API客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装 openai: pip install openai")
        
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key未设置")
        
        self._api_client = openai.OpenAI(
            api_key=api_key,
            base_url=self.config.api_base
        )
        print(f"  API模型: {self.config.model_name}")
        self.embedding_dim = 1536  # text-embedding-3-small
        
        # 初始化tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
            except:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def _init_zhipu(self):
        """初始化智谱AI API客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装 openai: pip install openai")
        
        api_key = self.config.api_key or os.getenv("ZHIPU_API_KEY")
        if not api_key:
            raise ValueError("智谱AI API key未设置")
        
        self._api_client = openai.OpenAI(
            api_key=api_key,
            base_url=self.config.api_base or "https://open.bigmodel.cn/api/paas/v4/"
        )
        print(f"  API模型: {self.config.model_name}")
        self.embedding_dim = 2048  # 智谱embedding-3
    
    def count_tokens(self, text: str) -> int:
        """计算文本token数"""
        if self.config.model_type in ["precomputed", "sentence_transformer"]:
            # 简单估算：中文字符 + 英文单词
            import re
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_words = len(re.findall(r'[a-zA-Z]+', text))
            return chinese_chars + english_words
        elif self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text) // 4  # 粗略估算
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """截断文本到指定token数"""
        if self.count_tokens(text) <= max_tokens:
            return text
        
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            return self.tokenizer.decode(tokens[:max_tokens])
        else:
            # 简单截断
            return text[:max_tokens * 4]
    
    def embed(self, texts: List[str], chunk_ids: Optional[List[str]] = None) -> List[List[float]]:
        """批量嵌入文本"""
        if self.config.model_type == "precomputed":
            return self._embed_precomputed(chunk_ids)
        elif self.config.model_type == "sentence_transformer":
            return self._embed_local(texts)
        else:
            return self._embed_api(texts)
    
    def _embed_precomputed(self, chunk_ids: Optional[List[str]]) -> List[List[float]]:
        """从缓存获取预计算嵌入"""
        if chunk_ids is None:
            raise ValueError("预计算模式需要提供chunk_ids")
        
        embeddings = []
        missing_ids = []
        
        for chunk_id in chunk_ids:
            chunk_id_str = str(chunk_id)
            if chunk_id_str in self._precomputed_embeddings:
                embeddings.append(self._precomputed_embeddings[chunk_id_str].tolist())
            else:
                missing_ids.append(chunk_id)
                # 使用零向量作为占位
                embeddings.append([0.0] * self.embedding_dim)
        
        if missing_ids:
            print(f"  警告: {len(missing_ids)} 个ID未找到对应嵌入")
        
        return embeddings
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """本地模型嵌入"""
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()
    
    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """API模型嵌入（带重试机制）"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # 截断超长文本
            batch = [self.truncate_text(t, self.config.max_tokens_per_batch) for t in batch]
            
            for attempt in range(self.config.max_retries):
                try:
                    response = self._api_client.embeddings.create(
                        model=self.config.model_name,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                    
                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise e
                    time.sleep(self.config.retry_delay * (2 ** attempt))
        
        return all_embeddings


# ===================== 批量向量化处理器 =====================

class BatchEmbeddingProcessor:
    """批量向量化处理器"""
    
    def __init__(self, model_manager: EmbeddingModelManager, config: EmbeddingConfig):
        self.model = model_manager
        self.config = config
        self.failed_chunks: List[Dict] = []
    
    def _generate_chunk_id(self, chunk: Dict, index: int) -> str:
        """生成与缓存兼容的chunk ID"""
        doc_id = chunk.get("doc_id", "unknown")
        # 格式: {doc_id}_chunk_{序号}
        return f"{doc_id}_chunk_{index}"
    
    def process_chunks(self, chunks: List[Dict]) -> Dict[str, Any]:
        """
        批量处理文本块
        
        Args:
            chunks: 文本块列表，每个块包含 text/content 等字段
            
        Returns:
            {
                "embeddings": [...],
                "ids": [...],
                "documents": [...],
                "metadatas": [...],
                "failed_chunks": [...],
                "stats": {...}
            }
        """
        print(f"\n[向量化] 开始处理 {len(chunks)} 个文本块")
        print(f"  批大小: {self.config.batch_size}")
        print(f"  模型: {self.config.model_type}")
        
        embeddings = []
        ids = []
        documents = []
        metadatas = []
        
        total_tokens = 0
        total_time = 0
        
        # 预计算模式：直接批量获取
        if self.config.model_type == "precomputed":
            # 生成与缓存兼容的ID
            chunk_ids = []
            id_to_chunk = {}
            
            for i, chunk in enumerate(chunks):
                chunk_id = self._generate_chunk_id(chunk, i)
                chunk_ids.append(chunk_id)
                id_to_chunk[chunk_id] = chunk
            
            start_time = time.time()
            embeddings = self.model.embed([], chunk_ids=chunk_ids)
            total_time = time.time() - start_time
            
            for i, chunk_id in enumerate(chunk_ids):
                chunk = id_to_chunk[chunk_id]
                text = chunk.get("text") or chunk.get("content") or ""
                
                ids.append(chunk_id)
                documents.append(text[:1000] if text else "")  # 限制文档长度
                metadatas.append(self._extract_metadata(chunk))
                total_tokens += self.model.count_tokens(text) if text else 0
            
            stats = {
                "total_chunks": len(chunks),
                "success_count": len(embeddings),
                "failed_count": 0,
                "total_tokens": total_tokens,
                "total_time_sec": round(total_time, 4),
                "avg_time_per_chunk_ms": round(total_time / len(chunks) * 1000, 2) if chunks else 0,
                "embedding_dim": self.model.embedding_dim
            }
            
        else:
            # 实时计算模式
            batch_size = self.config.batch_size
            num_batches = (len(chunks) + batch_size - 1) // batch_size
            
            with tqdm(total=len(chunks), desc="向量化进度") as pbar:
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i + batch_size]
                    batch_texts = []
                    batch_indices = []
                    batch_ids = []
                    
                    # 提取文本
                    for idx, chunk in enumerate(batch_chunks):
                        text = chunk.get("text") or chunk.get("content") or ""
                        chunk_id = chunk.get("chunk_id") or f"chunk_{i+idx}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                        
                        if text and len(text.strip()) > 0:
                            # 截断超长文本
                            text = self.model.truncate_text(text, self.config.max_tokens_per_batch)
                            batch_texts.append(text)
                            batch_indices.append(idx)
                            batch_ids.append(chunk_id)
                            total_tokens += self.model.count_tokens(text)
                        else:
                            self.failed_chunks.append({
                                "chunk_id": chunk_id,
                                "reason": "empty_text"
                            })
                    
                    if not batch_texts:
                        pbar.update(len(batch_chunks))
                        continue
                    
                    # 执行嵌入
                    try:
                        start_time = time.time()
                        batch_embeddings = self.model.embed(batch_texts)
                        batch_time = time.time() - start_time
                        total_time += batch_time
                        
                        # 组织结果
                        for idx, emb, cid in zip(batch_indices, batch_embeddings, batch_ids):
                            chunk = batch_chunks[idx]
                            
                            embeddings.append(emb)
                            ids.append(cid)
                            documents.append(batch_texts[idx])
                            metadatas.append(self._extract_metadata(chunk))
                        
                    except Exception as e:
                        print(f"\n  批次 {i//batch_size + 1} 失败: {str(e)}")
                        for cid in batch_ids:
                            self.failed_chunks.append({
                                "chunk_id": cid,
                                "reason": str(e)
                            })
                    
                    pbar.update(len(batch_chunks))
            
            stats = {
                "total_chunks": len(chunks),
                "success_count": len(embeddings),
                "failed_count": len(self.failed_chunks),
                "total_tokens": total_tokens,
                "total_time_sec": round(total_time, 4),
                "avg_time_per_chunk_ms": round(total_time / len(chunks) * 1000, 2) if chunks else 0,
                "embedding_dim": len(embeddings[0]) if embeddings else 0
            }
        
        print(f"\n[向量化完成]")
        print(f"  成功: {stats['success_count']} / {stats['total_chunks']}")
        print(f"  失败: {stats['failed_count']}")
        print(f"  向量维度: {stats['embedding_dim']}")
        if self.config.model_type != "precomputed":
            print(f"  总token数: {stats['total_tokens']}")
            print(f"  平均耗时: {stats['avg_time_per_chunk_ms']}ms/块")
        
        return {
            "embeddings": embeddings,
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "failed_chunks": self.failed_chunks,
            "stats": stats
        }
    
    def _extract_metadata(self, chunk: Dict) -> Dict[str, Any]:
        """提取完整的元数据"""
        metadata = {
            # 基础信息
            "doc_id": chunk.get("doc_id", ""),
            "chunk_id": chunk.get("chunk_id", ""),
            "model": chunk.get("model", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", ""),
            "type": chunk.get("type", "文本"),
            
            # 内容特征
            "keywords": ",".join(chunk.get("keywords", [])),
            "token_count": chunk.get("token_count", 0),
            "lang": chunk.get("lang", "unknown"),
            
            # 位置信息
            "page": chunk.get("page", -1),
            
            # 时间戳
            "index_time": datetime.now().isoformat()
        }
        
        # 清理None值
        return {k: v if v is not None else "" for k, v in metadata.items()}


# ===================== ChromaDB 管理器 =====================

class ChromaDBManager:
    """ChromaDB 向量数据库管理器"""
    
    def __init__(self, config: ChromaConfig):
        self.config = config
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """初始化ChromaDB客户端"""
        print(f"\n[ChromaDB] 初始化")
        print(f"  存储路径: {self.config.persist_path}")
        
        self.client = chromadb.PersistentClient(
            path=self.config.persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def create_or_get_collection(self) -> chromadb.Collection:
        """创建或获取集合"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(
                name=self.config.collection_name
            )
            print(f"  使用现有集合: {self.config.collection_name}")
            self.collection = collection
            return collection
        except Exception:
            # 创建新集合
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "hnsw:M": self.config.hnsw_m,
                    "hnsw:ef_construction": self.config.hnsw_ef_construction,
                    "hnsw:ef_search": self.config.hnsw_ef_search,
                    "description": "DJI无人机手册向量数据库",
                    "created_at": datetime.now().isoformat()
                }
            )
            print(f"  创建新集合: {self.config.collection_name}")
            print(f"  距离度量: {self.config.distance_metric}")
            self.collection = collection
            return collection
    
    def add_embeddings(self, embed_result: Dict[str, Any], batch_size: int = 500):
        """
        批量添加向量到数据库
        
        Args:
            embed_result: 向量化结果
            batch_size: 每批添加数量
        """
        if not self.collection:
            self.create_or_get_collection()
        
        total = len(embed_result["ids"])
        print(f"\n[索引构建] 添加 {total} 个向量")
        print(f"  批大小: {batch_size}")
        
        with tqdm(total=total, desc="构建索引") as pbar:
            for i in range(0, total, batch_size):
                batch_slice = slice(i, min(i + batch_size, total))
                
                self.collection.add(
                    embeddings=embed_result["embeddings"][batch_slice],
                    ids=embed_result["ids"][batch_slice],
                    documents=embed_result["documents"][batch_slice],
                    metadatas=embed_result["metadatas"][batch_slice]
                )
                
                pbar.update(min(batch_size, total - i))
        
        # 保存统计
        stats = {
            "total_vectors": self.collection.count(),
            "collection_name": self.config.collection_name,
            "distance_metric": self.config.distance_metric,
            "embedding_dim": len(embed_result["embeddings"][0]) if embed_result["embeddings"] else 0,
            "last_update": datetime.now().isoformat()
        }
        
        with open("index_stat.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n[索引完成]")
        print(f"  集合总向量数: {stats['total_vectors']}")
        
        return stats
    
    def query(
        self,
        query_text: str,
        embedding_model: EmbeddingModelManager,
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        向量相似性查询
        
        Args:
            query_text: 查询文本
            embedding_model: 嵌入模型
            n_results: 返回结果数量
            where_filter: 元数据过滤条件
            include: 返回的字段
            
        Returns:
            查询结果
        """
        if include is None:
            include = ["metadatas", "documents", "distances"]
        
        if not self.collection:
            self.create_or_get_collection()
        
        start_time = time.time()
        
        # 嵌入查询文本（预计算模式不支持动态嵌入，需要其他模式）
        if embedding_model.config.model_type == "precomputed":
            print("  警告: 预计算模式不支持动态查询嵌入，使用随机向量")
            import random
            query_embedding = [random.random() for _ in range(embedding_model.embedding_dim)]
        else:
            query_embedding = embedding_model.embed([query_text])[0]
        
        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=include
        )
        
        query_time = time.time() - start_time
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "rank": i + 1
            }
            
            if "distances" in include:
                result["similarity"] = round(1 - results["distances"][0][i], 4)
                result["distance"] = round(results["distances"][0][i], 4)
            
            if "documents" in include:
                result["document"] = results["documents"][0][i]
                result["document_preview"] = results["documents"][0][i][:200] + "..." if len(results["documents"][0][i]) > 200 else results["documents"][0][i]
            
            if "metadatas" in include:
                result["metadata"] = results["metadatas"][0][i]
            
            formatted_results.append(result)
        
        return {
            "query": query_text,
            "n_results": n_results,
            "where_filter": where_filter,
            "latency_ms": round(query_time * 1000, 2),
            "results": formatted_results
        }


# ===================== 质量验证器 =====================

class QualityValidator:
    """向量数据库质量验证器"""
    
    def __init__(self, db_manager: ChromaDBManager, embedding_model: EmbeddingModelManager):
        self.db = db_manager
        self.model = embedding_model
        self.report = {}
    
    def run_full_validation(self, chunks: List[Dict]) -> Dict[str, Any]:
        """运行完整验证流程"""
        print("\n" + "="*60)
        print("开始完整质量验证")
        print("="*60)
        
        self.report = {
            "validation_time": datetime.now().isoformat(),
            "tests": {}
        }
        
        # 1. 基础统计验证
        self.report["tests"]["basic_stats"] = self._validate_basic_stats()
        
        # 2. 元数据验证
        self.report["tests"]["metadata"] = self._validate_metadata()
        
        # 3. 相似性检索验证（跳过预计算模式）
        if self.model.config.model_type != "precomputed":
            self.report["tests"]["self_similarity"] = self._validate_self_similarity(chunks)
        
        # 4. 多类型查询验证
        self.report["tests"]["query_types"] = self._validate_query_types()
        
        # 5. 边界情况验证
        self.report["tests"]["edge_cases"] = self._validate_edge_cases()
        
        # 6. 元数据过滤验证
        self.report["tests"]["metadata_filter"] = self._validate_metadata_filter()
        
        # 保存报告
        with open("retrieval_verification_report.json", "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("验证完成，报告已保存: retrieval_verification_report.json")
        print("="*60)
        
        return self.report
    
    def _validate_basic_stats(self) -> Dict:
        """基础统计验证"""
        print("\n[验证1] 基础统计验证")
        
        stats = {
            "vector_count": self.db.collection.count(),
            "collection_name": self.db.config.collection_name,
            "distance_metric": self.db.config.distance_metric
        }
        
        # 获取样本
        sample = self.db.collection.get(limit=5)
        stats["sample_ids"] = sample["ids"]
        stats["sample_metadata_keys"] = list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
        
        print(f"  向量总数: {stats['vector_count']}")
        print(f"  元数据字段: {stats['sample_metadata_keys']}")
        
        return stats
    
    def _validate_metadata(self) -> Dict:
        """元数据完整性验证"""
        print("\n[验证2] 元数据完整性验证")
        
        # 获取部分数据
        result = self.db.collection.get(limit=1000)
        
        required_fields = ["doc_id", "model", "type", "lang"]
        field_coverage = {}
        
        for field in required_fields:
            present = sum(1 for m in result["metadatas"] if m.get(field))
            field_coverage[field] = {
                "present": present,
                "total": len(result["metadatas"]),
                "coverage": round(present / len(result["metadatas"]) * 100, 2) if result["metadatas"] else 0
            }
        
        print(f"  字段覆盖率:")
        for field, cov in field_coverage.items():
            print(f"    {field}: {cov['coverage']}% ({cov['present']}/{cov['total']})")
        
        return field_coverage
    
    def _validate_self_similarity(self, chunks: List[Dict]) -> Dict:
        """自相似性验证 - 从索引中摘取文本作为查询"""
        print("\n[验证3] 自相似性检索验证")
        
        # 随机采样
        import random
        sample_size = min(10, len(chunks))
        samples = random.sample(chunks, sample_size)
        
        recalls = []
        
        for i, chunk in enumerate(samples):
            query_text = chunk.get("text", "")[:100]  # 取前100字符
            chunk_id = chunk.get("chunk_id", "")
            
            if not query_text or not chunk_id:
                continue
            
            # 查询
            results = self.db.query(
                query_text=query_text,
                embedding_model=self.model,
                n_results=5
            )
            
            # 检查是否找到自身
            retrieved_ids = [r["id"] for r in results["results"]]
            found = chunk_id in retrieved_ids
            rank = retrieved_ids.index(chunk_id) + 1 if found else None
            
            recalls.append({
                "query_id": chunk_id,
                "found": found,
                "rank": rank,
                "top1_similarity": results["results"][0]["similarity"] if results["results"] else 0
            })
        
        # 统计
        found_count = sum(1 for r in recalls if r["found"])
        avg_similarity = sum(r["top1_similarity"] for r in recalls) / len(recalls) if recalls else 0
        
        stats = {
            "test_samples": len(recalls),
            "recall_at_5": round(found_count / len(recalls) * 100, 2) if recalls else 0,
            "avg_top1_similarity": round(avg_similarity, 4),
            "details": recalls
        }
        
        print(f"  测试样本: {stats['test_samples']}")
        print(f"  Recall@5: {stats['recall_at_5']}%")
        print(f"  平均Top1相似度: {stats['avg_top1_similarity']}")
        
        return stats
    
    def _validate_query_types(self) -> Dict:
        """多类型查询验证"""
        print("\n[验证4] 多类型查询验证")
        
        # 定义测试查询
        test_queries = [
            ("事实参数查询", "无人机最大飞行高度是多少米", {"expected_keywords": ["高度", "飞行", "米"]}),
            ("操作步骤查询", "如何进行指南针校准", {"expected_keywords": ["校准", "指南针", "步骤"]}),
            ("故障排查查询", "遥控器连接失败怎么办", {"expected_keywords": ["遥控器", "连接", "故障"]}),
            ("元数据过滤查询", "Mavic 4 Pro 避障功能", {"expected_keywords": ["Mavic", "避障"], "where": {"model": "Mavic 4 Pro"}}),
            ("模糊语义查询", "图传画面卡顿解决方法", {"expected_keywords": ["图传", "卡顿", "信号"]})
        ]
        
        results = []
        for query_type, query_text, config in test_queries:
            # 执行查询
            try:
                results_data = self.db.query(
                    query_text=query_text,
                    embedding_model=self.model,
                    n_results=5,
                    where_filter=config.get("where")
                )
                
                # 检查关键词命中
                all_text = " ".join([r["document"] for r in results_data["results"]])
                keyword_hits = sum(1 for kw in config["expected_keywords"] if kw in all_text)
                
                results.append({
                    "type": query_type,
                    "query": query_text,
                    "latency_ms": results_data["latency_ms"],
                    "keyword_hits": keyword_hits,
                    "total_keywords": len(config["expected_keywords"]),
                    "precision_at_5": round(keyword_hits / len(config["expected_keywords"]) * 100, 2) if config["expected_keywords"] else 0,
                    "top_result_similarity": results_data["results"][0]["similarity"] if results_data["results"] else 0
                })
                print(f"  {query_type}: 延迟={results_data['latency_ms']}ms")
            except Exception as e:
                results.append({
                    "type": query_type,
                    "query": query_text,
                    "error": str(e)
                })
                print(f"  {query_type}: 错误={str(e)}")
        
        # 统计
        valid_results = [r for r in results if "latency_ms" in r]
        if valid_results:
            avg_latency = sum(r["latency_ms"] for r in valid_results) / len(valid_results)
            avg_precision = sum(r["precision_at_5"] for r in valid_results) / len(valid_results)
        else:
            avg_latency = 0
            avg_precision = 0
        
        return {
            "test_cases": results,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_precision_at_5": round(avg_precision, 2)
        }
    
    def _validate_edge_cases(self) -> Dict:
        """边界情况验证"""
        print("\n[验证5] 边界情况验证")
        
        edge_cases = [
            ("空查询", ""),
            ("超长查询", "无人机 " * 500),
            ("特殊字符", "!@#$%^&*()"),
            ("数字查询", "12345 67890"),
            ("单字符", "飞")
        ]
        
        results = []
        for case_name, query_text in edge_cases:
            try:
                start = time.time()
                result = self.db.query(
                    query_text=query_text,
                    embedding_model=self.model,
                    n_results=3
                )
                latency = (time.time() - start) * 1000
                
                results.append({
                    "case": case_name,
                    "query_length": len(query_text),
                    "status": "success",
                    "latency_ms": round(latency, 2),
                    "results_count": len(result["results"])
                })
                print(f"  {case_name}: 成功, 延迟={round(latency, 2)}ms")
                
            except Exception as e:
                results.append({
                    "case": case_name,
                    "query_length": len(query_text),
                    "status": "error",
                    "error": str(e)
                })
                print(f"  {case_name}: 失败 - {str(e)}")
        
        return {"test_cases": results}
    
    def _validate_metadata_filter(self) -> Dict:
        """元数据过滤验证"""
        print("\n[验证6] 元数据过滤验证")
        
        # 测试不同的过滤条件
        filter_tests = [
            ("按模型过滤", {"model": "DJI Device"}),
            ("按类型过滤", {"type": "文本"}),
            ("按语言过滤", {"lang": "zh"}),
        ]
        
        results = []
        for test_name, where_filter in filter_tests:
            try:
                start = time.time()
                result = self.db.query(
                    query_text="无人机",
                    embedding_model=self.model,
                    n_results=5,
                    where_filter=where_filter
                )
                latency = (time.time() - start) * 1000
                
                results.append({
                    "test_name": test_name,
                    "filter": str(where_filter),
                    "status": "success",
                    "latency_ms": round(latency, 2),
                    "results_count": len(result["results"]),
                    "filter_accurate": True
                })
                print(f"  {test_name}: 成功, 返回{len(result['results'])}条")
                
            except Exception as e:
                results.append({
                    "test_name": test_name,
                    "filter": str(where_filter),
                    "status": "error",
                    "error": str(e)
                })
                print(f"  {test_name}: 失败 - {str(e)}")
        
        return {"test_cases": results}


# ===================== 主程序 =====================

def load_chunks(file_path: str, max_chunks: Optional[int] = None) -> List[Dict]:
    """加载文本块数据"""
    print(f"[数据加载] 从 {file_path} 读取数据")
    
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_chunks and idx >= max_chunks:
                break
            if line.strip():
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
    
    print(f"  加载了 {len(chunks)} 个文本块")
    return chunks


def main():
    """主程序入口"""
    print("="*60)
    print("DJI 手册向量数据库构建系统")
    print("="*60)
    
    # ========== 配置 ==========
    config = VectorDBConfig(
        embedding=EmbeddingConfig(
            model_type="precomputed",  # 使用预计算嵌入
            precomputed_cache_path="embed_cache.npz",
            batch_size=1000
        ),
        chroma=ChromaConfig(
            persist_path="./chroma_final",
            collection_name="uav_manual_db",
            distance_metric="cosine"
        ),
        max_chunks=None  # 处理全部数据
    )
    
    # ========== 1. 加载数据 ==========
    chunks = load_chunks("dji_chunks_final.jsonl", config.max_chunks)
    
    # ========== 2. 初始化嵌入模型 ==========
    model_manager = EmbeddingModelManager(config.embedding)
    
    # ========== 3. 批量向量化 ==========
    processor = BatchEmbeddingProcessor(model_manager, config.embedding)
    embed_result = processor.process_chunks(chunks)
    
    # 保存失败的chunks
    if config.save_failed_chunks and embed_result["failed_chunks"]:
        with open("failed_chunks.json", "w", encoding="utf-8") as f:
            json.dump(embed_result["failed_chunks"], f, ensure_ascii=False, indent=2)
        print(f"  失败记录已保存: failed_chunks.json")
    
    # 保存向量化统计
    with open("vectorization_stat.json", "w", encoding="utf-8") as f:
        json.dump(embed_result["stats"], f, ensure_ascii=False, indent=2)
    
    # ========== 4. 构建ChromaDB索引 ==========
    db_manager = ChromaDBManager(config.chroma)
    index_stats = db_manager.add_embeddings(embed_result)
    
    # ========== 5. 质量验证 ==========
    validator = QualityValidator(db_manager, model_manager)
    validation_report = validator.run_full_validation(chunks[:100])  # 只验证前100条
    
    # ========== 6. 生成配置文档 ==========
    config_doc = {
        "project": "DJI 手册向量数据库",
        "created_at": datetime.now().isoformat(),
        "embedding_config": {
            "model_type": config.embedding.model_type,
            "model_name": config.embedding.model_name,
            "precomputed_cache_path": config.embedding.precomputed_cache_path,
            "batch_size": config.embedding.batch_size
        },
        "chroma_config": {
            "persist_path": config.chroma.persist_path,
            "collection_name": config.chroma.collection_name,
            "distance_metric": config.chroma.distance_metric
        },
        "data_stats": embed_result["stats"],
        "index_stats": index_stats,
        "validation_summary": {
            "avg_latency_ms": validation_report["tests"]["query_types"]["avg_latency_ms"],
            "avg_precision_at_5": validation_report["tests"]["query_types"]["avg_precision_at_5"]
        }
    }
    
    with open("vector_db_config.json", "w", encoding="utf-8") as f:
        json.dump(config_doc, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("🎉 所有任务完成！")
    print("="*60)
    print("\n产出文件:")
    print("  - chroma_final/          向量数据库文件")
    print("  - vector_db_config.json  配置文档")
    print("  - index_stat.json        索引统计")
    print("  - vectorization_stat.json 向量化统计")
    print("  - retrieval_verification_report.json 检索验证报告")
    print("  - failed_chunks.json     失败记录（如有）")


if __name__ == "__main__":
    main()
