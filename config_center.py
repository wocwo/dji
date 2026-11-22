"""
统一配置中心 (Config Center)

功能：
1. 将所有模块的配置参数集中管理
2. 使用 YAML 配置文件
3. 支持不同环境（开发 / 测试 / 生产）的配置切换
4. 配置热加载与验证
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger(__name__)


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class RetrievalConfig:
    """检索模块配置"""
    chroma_db_path: str = "./chroma_final"
    collection_name: str = "uav_manual_db"
    embedding_model_type: str = "precomputed"
    embedding_cache_path: str = "embed_cache.npz"
    multipath_top_k: int = 50
    final_top_k: int = 10
    reranker_type: str = "simple"
    reranker_threshold: float = 0.3


@dataclass
class LLMConfig:
    """LLM模块配置"""
    backend: str = "ollama"
    model_name: str = "deepseek-r1:7b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    timeout: float = 120.0
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 2048
    frequency_penalty: float = 0.0
    system_prompt: Optional[str] = None
    # vLLM 特有
    vllm_mode: str = "server"
    device: str = "cuda"


@dataclass
class PipelineConfig:
    """流水线配置"""
    max_context_chars: int = 8000
    score_threshold: float = 0.3
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_post_processing: bool = True
    add_citations: bool = True
    add_disclaimer: bool = True
    max_retries: int = 2
    retry_delay: float = 1.0


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# =============================================================================
# 配置中心
# =============================================================================

class ConfigCenter:
    """
    统一配置中心
    
    管理所有模块的配置，支持从 YAML 文件加载和环境切换
    
    Usage:
        config = ConfigCenter("configs/dev.yaml")
        
        # 获取检索配置
        retrieval_cfg = config.retrieval
        
        # 获取LLM配置
        llm_cfg = config.llm
        
        # 获取完整配置字典
        full_cfg = config.to_dict()
    """

    # 默认配置目录
    DEFAULT_CONFIG_DIR = "configs"
    
    # 环境到配置文件的映射
    ENV_CONFIG_MAP = {
        "dev": "dev.yaml",
        "test": "test.yaml",
        "prod": "prod.yaml",
        "development": "dev.yaml",
        "testing": "test.yaml",
        "production": "prod.yaml"
    }

    def __init__(self, config_path: Optional[str] = None, env: Optional[str] = None):
        """
        初始化配置中心
        
        Args:
            config_path: 配置文件路径，如果为None则根据env自动选择
            env: 环境名称 (dev/test/prod)，如果config_path为None则使用
        """
        self._raw_config: Dict[str, Any] = {}
        self._config_path: Optional[str] = None
        
        # 确定配置文件路径
        if config_path:
            self._config_path = config_path
        elif env:
            self._config_path = self._get_config_path_by_env(env)
        else:
            # 尝试从环境变量读取
            env_from_var = os.environ.get("RAG_ENV", "dev")
            self._config_path = self._get_config_path_by_env(env_from_var)
        
        # 加载配置
        self._load_config()
        
        # 解析为结构化配置
        self._parse_config()
        
        logger.info(f"[Config] 配置加载完成: {self._config_path}")

    def _get_config_path_by_env(self, env: str) -> str:
        """根据环境获取配置文件路径"""
        filename = self.ENV_CONFIG_MAP.get(env.lower(), "dev.yaml")
        return os.path.join(self.DEFAULT_CONFIG_DIR, filename)

    def _load_config(self):
        """从 YAML 文件加载配置"""
        if not os.path.exists(self._config_path):
            logger.warning(
                f"[Config] 配置文件不存在: {self._config_path}，使用默认配置"
            )
            self._raw_config = self._default_config()
            return
        
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._raw_config = yaml.safe_load(f) or {}
            logger.info(f"[Config] 已加载配置: {self._config_path}")
        except Exception as e:
            logger.error(f"[Config] 配置加载失败: {e}，使用默认配置")
            self._raw_config = self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "retrieval": {
                "chroma_db_path": "./chroma_final",
                "collection_name": "uav_manual_db",
                "embedding_model_type": "precomputed",
                "embedding_cache_path": "embed_cache.npz",
                "multipath_top_k": 50,
                "final_top_k": 10,
                "reranker_type": "simple",
                "reranker_threshold": 0.3
            },
            "llm": {
                "backend": "ollama",
                "model_name": "deepseek-r1:7b",
                "base_url": "http://localhost:11434",
                "timeout": 120.0,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "frequency_penalty": 0.0
            },
            "pipeline": {
                "max_context_chars": 8000,
                "score_threshold": 0.3,
                "enable_logging": True,
                "log_level": "INFO",
                "enable_post_processing": True,
                "add_citations": True,
                "add_disclaimer": True,
                "max_retries": 2,
                "retry_delay": 1.0
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def _parse_config(self):
        """解析配置为结构化对象"""
        # 检索配置
        r = self._raw_config.get("retrieval", {})
        self.retrieval = RetrievalConfig(
            chroma_db_path=r.get("chroma_db_path", "./chroma_final"),
            collection_name=r.get("collection_name", "uav_manual_db"),
            embedding_model_type=r.get("embedding_model_type", "precomputed"),
            embedding_cache_path=r.get("embedding_cache_path", "embed_cache.npz"),
            multipath_top_k=r.get("multipath_top_k", 50),
            final_top_k=r.get("final_top_k", 10),
            reranker_type=r.get("reranker_type", "simple"),
            reranker_threshold=r.get("reranker_threshold", 0.3)
        )
        
        # LLM配置
        l = self._raw_config.get("llm", {})
        self.llm = LLMConfig(
            backend=l.get("backend", "ollama"),
            model_name=l.get("model_name", "deepseek-r1:7b"),
            base_url=l.get("base_url", "http://localhost:11434"),
            api_key=l.get("api_key"),
            timeout=l.get("timeout", 120.0),
            temperature=l.get("temperature", 0.3),
            top_p=l.get("top_p", 0.9),
            max_tokens=l.get("max_tokens", 2048),
            frequency_penalty=l.get("frequency_penalty", 0.0),
            system_prompt=l.get("system_prompt"),
            vllm_mode=l.get("vllm_mode", "server"),
            device=l.get("device", "cuda")
        )
        
        # 流水线配置
        p = self._raw_config.get("pipeline", {})
        self.pipeline = PipelineConfig(
            max_context_chars=p.get("max_context_chars", 8000),
            score_threshold=p.get("score_threshold", 0.3),
            enable_logging=p.get("enable_logging", True),
            log_level=p.get("log_level", "INFO"),
            enable_post_processing=p.get("enable_post_processing", True),
            add_citations=p.get("add_citations", True),
            add_disclaimer=p.get("add_disclaimer", True),
            max_retries=p.get("max_retries", 2),
            retry_delay=p.get("retry_delay", 1.0)
        )
        
        # 日志配置
        lg = self._raw_config.get("logging", {})
        self.logging = LoggingConfig(
            level=lg.get("level", "INFO"),
            format=lg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=lg.get("file"),
            max_bytes=lg.get("max_bytes", 10 * 1024 * 1024),
            backup_count=lg.get("backup_count", 5)
        )

    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._parse_config()
        logger.info("[Config] 配置已重新加载")

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "retrieval": {
                "chroma_db_path": self.retrieval.chroma_db_path,
                "collection_name": self.retrieval.collection_name,
                "embedding_model_type": self.retrieval.embedding_model_type,
                "embedding_cache_path": self.retrieval.embedding_cache_path,
                "multipath_top_k": self.retrieval.multipath_top_k,
                "final_top_k": self.retrieval.final_top_k,
                "reranker_type": self.retrieval.reranker_type,
                "reranker_threshold": self.retrieval.reranker_threshold
            },
            "llm": {
                "backend": self.llm.backend,
                "model_name": self.llm.model_name,
                "base_url": self.llm.base_url,
                "timeout": self.llm.timeout,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "frequency_penalty": self.llm.frequency_penalty,
                "vllm_mode": self.llm.vllm_mode,
                "device": self.llm.device
            },
            "pipeline": {
                "max_context_chars": self.pipeline.max_context_chars,
                "score_threshold": self.pipeline.score_threshold,
                "enable_logging": self.pipeline.enable_logging,
                "log_level": self.pipeline.log_level,
                "enable_post_processing": self.pipeline.enable_post_processing,
                "add_citations": self.pipeline.add_citations,
                "add_disclaimer": self.pipeline.add_disclaimer,
                "max_retries": self.pipeline.max_retries,
                "retry_delay": self.pipeline.retry_delay
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "file": self.logging.file,
                "max_bytes": self.logging.max_bytes,
                "backup_count": self.logging.backup_count
            }
        }

    def save(self, path: Optional[str] = None):
        """
        保存当前配置到 YAML 文件
        
        Args:
            path: 保存路径，默认覆盖原文件
        """
        save_path = path or self._config_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
            logger.info(f"[Config] 配置已保存: {save_path}")
        except Exception as e:
            logger.error(f"[Config] 配置保存失败: {e}")
            raise

    @classmethod
    def create_env_config(
        cls,
        env: str,
        output_dir: str = "configs"
    ) -> str:
        """
        创建环境配置文件
        
        Args:
            env: 环境名称
            output_dir: 输出目录
            
        Returns:
            创建的文件路径
        """
        filename = cls.ENV_CONFIG_MAP.get(env.lower(), f"{env}.yaml")
        filepath = os.path.join(output_dir, filename)
        
        config = cls._template_config(env)
        
        os.makedirs(output_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        
        logger.info(f"[Config] 已创建 {env} 环境配置: {filepath}")
        return filepath

    @staticmethod
    def _template_config(env: str) -> Dict[str, Any]:
        """根据环境生成配置模板"""
        base = {
            "retrieval": {
                "chroma_db_path": "./chroma_final",
                "collection_name": "uav_manual_db",
                "embedding_model_type": "precomputed",
                "embedding_cache_path": "embed_cache.npz",
                "multipath_top_k": 50,
                "final_top_k": 10,
                "reranker_type": "simple",
                "reranker_threshold": 0.3
            },
            "llm": {
                "backend": "ollama",
                "model_name": "deepseek-r1:7b",
                "base_url": "http://localhost:11434",
                "timeout": 120.0,
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048,
                "frequency_penalty": 0.0
            },
            "pipeline": {
                "max_context_chars": 8000,
                "score_threshold": 0.3,
                "enable_logging": True,
                "log_level": "INFO",
                "enable_post_processing": True,
                "add_citations": True,
                "add_disclaimer": True,
                "max_retries": 2,
                "retry_delay": 1.0
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        if env == "prod":
            base["llm"]["backend"] = "vllm"
            base["llm"]["model_name"] = "Qwen/Qwen2-7B-Instruct-AWQ"
            base["llm"]["base_url"] = "http://localhost:8000"
            base["llm"]["vllm_mode"] = "server"
            base["retrieval"]["reranker_type"] = "cross_encoder"
            base["pipeline"]["max_retries"] = 3
            base["pipeline"]["log_level"] = "WARNING"
            base["logging"]["level"] = "WARNING"
            base["logging"]["file"] = "logs/rag_pipeline.log"
        elif env == "test":
            base["llm"]["model_name"] = "qwen2:1.8b"
            base["llm"]["timeout"] = 60.0
            base["pipeline"]["max_retries"] = 1
            base["pipeline"]["log_level"] = "DEBUG"
            base["logging"]["level"] = "DEBUG"
        
        return base


# =============================================================================
# 便捷函数
# =============================================================================

def get_config(env: Optional[str] = None) -> ConfigCenter:
    """
    便捷函数：获取配置中心实例
    
    Args:
        env: 环境名称，None则从环境变量读取
        
    Returns:
        ConfigCenter实例
    """
    return ConfigCenter(env=env)


def setup_logging(config: Optional[ConfigCenter] = None):
    """
    根据配置设置日志
    
    Args:
        config: 配置中心实例，None则自动加载
    """
    if config is None:
        config = get_config()
    
    log_cfg = config.logging
    
    handlers = [logging.StreamHandler()]
    
    if log_cfg.file:
        os.makedirs(os.path.dirname(log_cfg.file), exist_ok=True)
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(
            log_cfg.file,
            maxBytes=log_cfg.max_bytes,
            backupCount=log_cfg.backup_count,
            encoding='utf-8'
        ))
    
    logging.basicConfig(
        level=getattr(logging, log_cfg.level.upper(), logging.INFO),
        format=log_cfg.format,
        handlers=handlers
    )
    
    logger.info("[Config] 日志系统已初始化")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("统一配置中心")
    print("=" * 70)
    
    # 创建各环境配置
    print("\n创建环境配置文件...")
    ConfigCenter.create_env_config("dev")
    ConfigCenter.create_env_config("test")
    ConfigCenter.create_env_config("prod")
    
    print("\n加载开发环境配置:")
    config = ConfigCenter(env="dev")
    print(f"  LLM后端: {config.llm.backend}")
    print(f"  模型: {config.llm.model_name}")
    print(f"  重排序器: {config.retrieval.reranker_type}")
    print(f"  上下文长度: {config.pipeline.max_context_chars}")
    
    print("\n使用方式:")
    print("  from config_center import ConfigCenter, setup_logging")
    print("  config = ConfigCenter('configs/dev.yaml')")
    print("  setup_logging(config)")
