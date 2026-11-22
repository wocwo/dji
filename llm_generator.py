"""
本地LLM集成与生成模块 (LLM Generator)

功能：
1. 支持 vLLM 高性能 GPU 部署（方案A，推荐）
2. 支持 Ollama 部署（方案B）
3. 统一的生成接口，支持系统提示、JSON格式要求
4. 生成参数优化与输出后处理
"""

import re
import json
import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Generator
from dataclasses import dataclass, field
from datetime import datetime


logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    # JSON模式
    response_format: Optional[str] = None  # "json" 或 None
    # 流式输出
    stream: bool = False


@dataclass
class GenerationResult:
    """生成结果"""
    text: str
    raw_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    generation_time_ms: float = 0.0
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMHealthStatus:
    """LLM健康状态"""
    healthy: bool
    model_name: str
    latency_ms: float
    error_message: Optional[str] = None
    version: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None


# =============================================================================
# 系统提示词模板
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """你是大疆无人机产品说明书专业问答助手。
你的回答必须严格基于用户提供的检索上下文（大疆官方产品说明书内容）。
严禁注入任何外部知识、个人经验、网络信息或推测。
回答应准确、简洁、有条理。
"""

JSON_SYSTEM_PROMPT_APPEND = """
【输出格式要求】
你必须以JSON格式输出，JSON结构如下：
{
    "answer": "你的详细回答",
    "confidence": "high|medium|low",
    "sources": ["来源1", "来源2"],
    "disclaimer": "免责声明内容"
}
注意：
1. 只输出JSON，不要输出其他解释性文字
2. 确保JSON格式正确，键名用双引号
3. 如果信息不足，answer字段使用标准回答："根据当前检索到的大疆官方产品说明书内容，暂无相关信息，建议您查阅对应机型的最新用户手册或联系大疆官方客服获取准确信息。"
"""


# =============================================================================
# 抽象基类
# =============================================================================

class BaseLLMGenerator(ABC):
    """LLM生成器抽象基类"""

    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.default_config = default_config or GenerationConfig()
        self._initialized = False
        self._health_status: Optional[LLMHealthStatus] = None

    @abstractmethod
    def initialize(self) -> bool:
        """初始化并测试连接，返回是否成功"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """
        生成回答
        
        Args:
            prompt: 用户提示词
            config: 生成配置（覆盖默认配置）
            system_prompt: 系统提示词（覆盖默认系统提示词）
        
        Returns:
            GenerationResult对象
        """
        pass

    @abstractmethod
    def health_check(self) -> LLMHealthStatus:
        """健康检查"""
        pass

    def build_full_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        require_json: bool = False
    ) -> str:
        """
        构建完整提示词
        
        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            require_json: 是否要求JSON格式输出
            
        Returns:
            完整提示词
        """
        sp = system_prompt or self.system_prompt
        
        if require_json and "JSON" not in sp:
            sp = sp + "\n" + JSON_SYSTEM_PROMPT_APPEND
        
        # 使用 ChatML 风格格式
        full_prompt = f"<|im_start|>system\n{sp}<|im_end|>\n"
        full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n"
        full_prompt += "<|im_start|>assistant\n"
        
        return full_prompt

    def extract_json(self, text: str) -> Optional[Union[Dict, List]]:
        """
        从文本中提取 JSON 部分，并修复格式
        
        Args:
            text: LLM生成的原始文本
            
        Returns:
            解析后的JSON对象，失败返回None
        """
        # 1. 尝试直接解析
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # 2. 提取 ```json ... ``` 代码块
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(json_block_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        
        # 3. 提取花括号内容（最外层）
        brace_pattern = r'(\{[\s\S]*\})'
        matches = re.findall(brace_pattern, text)
        for match in matches:
            # 尝试修复常见格式问题
            fixed = self._fix_json_format(match)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
        
        # 4. 提取方括号内容
        bracket_pattern = r'(\[[\s\S]*\])'
        matches = re.findall(bracket_pattern, text)
        for match in matches:
            fixed = self._fix_json_format(match)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
        
        return None

    @staticmethod
    def _fix_json_format(text: str) -> str:
        """
        修复常见 JSON 格式问题
        
        - 添加缺失的逗号
        - 修复单引号为双引号
        - 移除尾随逗号
        """
        # 去除首尾空白
        text = text.strip()
        
        # 将单引号替换为双引号
        # 键名：'key' -> "key" (在 { [ , 后面，紧接着 :)
        def replace_key_quotes(match):
            return match.group(1) + '"' + match.group(2) + '"' + match.group(3)
        text = re.sub(r"([\{\[\,]\s*)'([^']+)'(\s*:\s*)", replace_key_quotes, text)
        
        # 字符串值：'value' -> "value" (在 : 后面)
        def replace_value_quotes(match):
            return match.group(1) + '"' + match.group(2) + '"'
        text = re.sub(r"(:\s*)'([^']*)'", replace_value_quotes, text)
        
        # 数组中的字符串值：, 'value' -> , "value"
        text = re.sub(r"([\,\[]\s*)'([^']*)'", replace_value_quotes, text)
        
        # 移除尾随逗号（对象和数组中的）
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # 修复缺失的逗号：在字符串/数字/布尔值后紧跟键名的情况
        text = re.sub(r'(\}|"|\d|true|false)\s+("[^"]+")\s*:', r'\1, \2:', text)
        
        return text

    def merge_config(
        self,
        override: Optional[Union[GenerationConfig, Dict[str, Any]]] = None
    ) -> GenerationConfig:
        """
        合并配置，override 覆盖默认值
        
        Args:
            override: GenerationConfig 实例或字典，包含要覆盖的参数
        
        Returns:
            合并后的 GenerationConfig
        """
        import copy
        base = copy.copy(self.default_config)
        base.stop_sequences = self.default_config.stop_sequences.copy()
        
        if override is None:
            return base
        
        # 统一转为字典处理
        if isinstance(override, GenerationConfig):
            override_dict = {
                k: v for k, v in vars(override).items()
                # 排除与默认 dataclass 值相同的字段，避免误覆盖
                if v != getattr(GenerationConfig(), k, None)
            }
        else:
            override_dict = override
        
        for key, value in override_dict.items():
            if hasattr(base, key):
                setattr(base, key, value)
        
        return base


# =============================================================================
# 方案 A: vLLM 生成器
# =============================================================================

class VLLMGenerator(BaseLLMGenerator):
    """
    vLLM 高性能 GPU 部署生成器
    
    支持两种模式：
    1. 本地模式：直接加载 AWQ 量化模型（需要 transformers + autoawq）
    2. 服务模式：通过 HTTP API 调用 vLLM 服务
    
    Args:
        model_name: HuggingFace模型名称或本地路径，如 "Qwen/Qwen2-7B-Instruct-AWQ"
        base_url: vLLM服务地址（服务模式下使用）
        api_key: API密钥（可选）
        timeout: 请求超时时间（秒）
        mode: "local" 或 "server"
        device: 本地模式下的运行设备
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct-AWQ",
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        mode: str = "server",
        device: str = "cuda",
        system_prompt: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, system_prompt, default_config)
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.mode = mode
        self.device = device
        
        # 本地模式组件
        self._local_model = None
        self._local_tokenizer = None
        self._sampling_params = None
        
        logger.info(f"[VLLM] 初始化: model={model_name}, mode={mode}")

    def initialize(self) -> bool:
        """初始化并测试连接"""
        try:
            if self.mode == "local":
                return self._init_local()
            else:
                return self._init_server()
        except Exception as e:
            logger.error(f"[VLLM] 初始化失败: {e}")
            self._initialized = False
            return False

    def _init_local(self) -> bool:
        """初始化本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"[VLLM] 加载本地模型: {self.model_name}")
            
            self._local_tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True
            )
            
            self._initialized = True
            logger.info("[VLLM] 本地模型加载成功")
            return True
            
        except ImportError as e:
            logger.error(f"[VLLM] 缺少依赖: {e}，请安装 transformers 和 autoawq")
            return False
        except Exception as e:
            logger.error(f"[VLLM] 本地模型加载失败: {e}")
            return False

    def _init_server(self) -> bool:
        """初始化并测试 vLLM 服务连接"""
        try:
            headers = self._get_headers()
            
            # 测试连接：获取模型列表
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()
            
            models = response.json().get("data", [])
            model_ids = [m.get("id") for m in models]
            
            if self.model_name not in model_ids and len(model_ids) > 0:
                logger.warning(
                    f"[VLLM] 模型 {self.model_name} 不在服务列表中，"
                    f"可用模型: {model_ids}"
                )
            
            self._initialized = True
            logger.info(f"[VLLM] 服务连接成功: {self.base_url}, 模型: {model_ids}")
            return True
            
        except requests.exceptions.ConnectionError:
            logger.error(f"[VLLM] 无法连接到服务: {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"[VLLM] 服务测试失败: {e}")
            return False

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """生成回答"""
        if not self._initialized:
            raise RuntimeError("生成器未初始化，请先调用 initialize()")
        
        cfg = self.merge_config(config)
        full_prompt = self.build_full_prompt(prompt, system_prompt, cfg.response_format == "json")
        
        start_time = time.time()
        
        try:
            if self.mode == "local":
                result = self._generate_local(full_prompt, cfg)
            else:
                result = self._generate_server(full_prompt, cfg)
            
            result.generation_time_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.error(f"[VLLM] 生成失败: {e}")
            raise

    def _generate_local(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """本地生成"""
        import torch
        
        inputs = self._local_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                pad_token_id=self._local_tokenizer.eos_token_id
            )
        
        generated_text = self._local_tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        prompt_tokens = inputs['input_ids'].shape[1]
        completion_tokens = outputs.shape[1] - prompt_tokens
        
        return GenerationResult(
            text=generated_text.strip(),
            raw_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop"
        )

    def _generate_server(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """服务端生成"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "stream": False
        }
        
        if config.stop_sequences:
            payload["stop"] = config.stop_sequences
        
        response = requests.post(
            f"{self.base_url}/v1/completions",
            headers=self._get_headers(),
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        choice = data.get("choices", [{}])[0]
        generated_text = choice.get("text", "")
        
        usage = data.get("usage", {})
        
        return GenerationResult(
            text=generated_text.strip(),
            raw_text=generated_text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=choice.get("finish_reason", "")
        )

    def health_check(self) -> LLMHealthStatus:
        """健康检查"""
        start_time = time.time()
        
        try:
            if self.mode == "server":
                response = requests.get(
                    f"{self.base_url}/v1/models",
                    headers=self._get_headers(),
                    timeout=10.0
                )
                response.raise_for_status()
                latency = (time.time() - start_time) * 1000
                
                models = response.json().get("data", [])
                version = models[0].get("id") if models else None
                
                self._health_status = LLMHealthStatus(
                    healthy=True,
                    model_name=self.model_name,
                    latency_ms=latency,
                    version=version
                )
            else:
                # 本地模式简单检查
                latency = (time.time() - start_time) * 1000
                healthy = self._local_model is not None and self._local_tokenizer is not None
                self._health_status = LLMHealthStatus(
                    healthy=healthy,
                    model_name=self.model_name,
                    latency_ms=latency
                )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._health_status = LLMHealthStatus(
                healthy=False,
                model_name=self.model_name,
                latency_ms=latency,
                error_message=str(e)
            )
        
        return self._health_status


# =============================================================================
# 方案 B: Ollama 生成器
# =============================================================================

class OllamaGenerator(BaseLLMGenerator):
    """
    Ollama 部署生成器
    
    通过 HTTP API 调用 Ollama 服务
    
    Args:
        model_name: Ollama模型名称，如 "deepseek-r1:7b", "qwen2:7b"
        base_url: Ollama服务地址
        timeout: 请求超时时间（秒）
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:7b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        system_prompt: Optional[str] = None,
        default_config: Optional[GenerationConfig] = None
    ):
        super().__init__(model_name, system_prompt, default_config)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        logger.info(f"[Ollama] 初始化: model={model_name}, url={base_url}")

    def initialize(self) -> bool:
        """初始化并测试连接"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10.0
            )
            response.raise_for_status()
            
            models = response.json().get("models", [])
            model_names = [m.get("name") for m in models]
            
            if self.model_name not in model_names:
                logger.warning(
                    f"[Ollama] 模型 {self.model_name} 未找到，"
                    f"可用模型: {model_names}"
                )
                logger.info(f"[Ollama] 尝试拉取模型: {self.model_name}")
                # 尝试拉取模型（异步，不等待完成）
                try:
                    requests.post(
                        f"{self.base_url}/api/pull",
                        json={"name": self.model_name},
                        timeout=5.0
                    )
                except requests.exceptions.Timeout:
                    logger.info("[Ollama] 拉取请求已发送，正在后台下载...")
            
            self._initialized = True
            logger.info(f"[Ollama] 连接成功，可用模型: {model_names}")
            return True
            
        except requests.exceptions.ConnectionError:
            logger.error(f"[Ollama] 无法连接到服务: {self.base_url}")
            return False
        except Exception as e:
            logger.error(f"[Ollama] 初始化失败: {e}")
            return False

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> GenerationResult:
        """生成回答"""
        if not self._initialized:
            raise RuntimeError("生成器未初始化，请先调用 initialize()")
        
        cfg = self.merge_config(config)
        sp = system_prompt or self.system_prompt
        
        # Ollama 使用 messages 格式更可靠
        messages = []
        if sp:
            messages.append({"role": "system", "content": sp})
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "num_predict": cfg.max_tokens,
                "frequency_penalty": cfg.frequency_penalty,
                "presence_penalty": cfg.presence_penalty,
                "stop": cfg.stop_sequences if cfg.stop_sequences else []
            }
        }
        
        # 如果要求JSON格式，通过系统提示词注入要求
        if cfg.response_format == "json":
            payload["format"] = "json"
        
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        generated_text = data.get("message", {}).get("content", "")
        
        # Ollama 不返回 token 数，需要估算
        prompt_tokens = self._estimate_tokens(prompt + sp if sp else prompt)
        completion_tokens = self._estimate_tokens(generated_text)
        
        generation_time = (time.time() - start_time) * 1000
        
        return GenerationResult(
            text=generated_text.strip(),
            raw_text=generated_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            generation_time_ms=generation_time,
            finish_reason=data.get("done_reason", ""),
            metadata={
                "eval_count": data.get("eval_count", 0),
                "eval_duration": data.get("eval_duration", 0),
                "load_duration": data.get("load_duration", 0)
            }
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（中文字符约1.5 tokens/字，英文约0.25 tokens/char）"""
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars * 0.25)

    def health_check(self) -> LLMHealthStatus:
        """健康检查"""
        start_time = time.time()
        
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10.0
            )
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000
            
            self._health_status = LLMHealthStatus(
                healthy=True,
                model_name=self.model_name,
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self._health_status = LLMHealthStatus(
                healthy=False,
                model_name=self.model_name,
                latency_ms=latency,
                error_message=str(e)
            )
        
        return self._health_status


# =============================================================================
# 统一工厂类
# =============================================================================

class LLMGenerator:
    """
    LLM生成器统一入口
    
    根据配置自动选择 vLLM 或 Ollama 后端
    
    Usage:
        # vLLM 方案（推荐）
        generator = LLMGenerator.create("vllm", model_name="Qwen/Qwen2-7B-Instruct-AWQ")
        
        # Ollama 方案
        generator = LLMGenerator.create("ollama", model_name="deepseek-r1:7b")
        
        # 生成
        result = generator.generate("Mavic 4 Pro怎么起飞？", config=GenerationConfig(temperature=0.3))
        print(result.text)
    """

    GENERATOR_REGISTRY = {
        "vllm": VLLMGenerator,
        "ollama": OllamaGenerator
    }

    @classmethod
    def create(
        cls,
        backend: str = "ollama",
        **kwargs
    ) -> BaseLLMGenerator:
        """
        创建生成器实例
        
        Args:
            backend: 后端类型 ("vllm" | "ollama")
            **kwargs: 传递给具体生成器的参数
            
        Returns:
            BaseLLMGenerator实例
        """
        backend = backend.lower()
        generator_class = cls.GENERATOR_REGISTRY.get(backend)
        
        if generator_class is None:
            raise ValueError(
                f"不支持的后端: {backend}，"
                f"可选: {list(cls.GENERATOR_REGISTRY.keys())}"
            )
        
        generator = generator_class(**kwargs)
        return generator

    @classmethod
    def list_backends(cls) -> List[str]:
        """列出支持的后端"""
        return list(cls.GENERATOR_REGISTRY.keys())


# =============================================================================
# 便捷函数
# =============================================================================

def create_generator(
    backend: str = "ollama",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMGenerator:
    """
    便捷函数：创建LLM生成器
    
    Args:
        backend: 后端类型
        model_name: 模型名称
        base_url: 服务地址
        **kwargs: 其他参数
        
    Returns:
        生成器实例
    """
    if model_name:
        kwargs["model_name"] = model_name
    if base_url:
        kwargs["base_url"] = base_url
    
    generator = LLMGenerator.create(backend, **kwargs)
    
    # 自动初始化
    success = generator.initialize()
    if not success:
        logger.warning(f"生成器初始化失败，后续调用可能出错")
    
    return generator


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("本地LLM集成与生成模块")
    print("=" * 70)
    print("\n支持的后端:")
    for b in LLMGenerator.list_backends():
        print(f"  - {b}")
    
    print("\n使用方式:")
    print("  from llm_generator import create_generator, GenerationConfig")
    print("  gen = create_generator('ollama', model_name='deepseek-r1:7b')")
    print("  result = gen.generate('Mavic 4 Pro怎么起飞？')")
    print("  print(result.text)")
    
    print("\n  # JSON输出")
    print("  cfg = GenerationConfig(response_format='json', temperature=0.3)")
    print("  result = gen.generate('...', config=cfg)")
    print("  json_data = gen.extract_json(result.text)")
