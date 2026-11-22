"""LLM 客户端：Ollama / OpenAI 兼容 / 脚本化 / 离线确定性策略。

统一接口只暴露 ``complete(messages, ...) -> LLMResponse``，运行时无需关心后端。
原生 function calling 通过 ``tools`` 参数传入，响应中的 ``tool_calls`` 由运行时处理；
文本协议（默认）则完全由提示词 + 解析器驱动，适配不支持工具调用的本地模型。
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import requests

from .config import LLMSettings
from .errors import LLMError
from .types import Message, ToolCall, Usage


# ---------------------------------------------------------------------------
# 响应
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """一次 LLM 调用的结果。"""

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "usage": self.usage.to_dict(),
            "model": self.model,
            "finish_reason": self.finish_reason,
        }


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """LLM 客户端抽象。"""

    def __init__(self, settings: LLMSettings):
        self.settings = settings

    @property
    def name(self) -> str:
        return self.settings.provider

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
    ) -> LLMResponse:
        """执行一次补全。"""

    def health(self) -> Dict[str, Any]:
        """健康检查，返回可序列化字典。"""
        return {"healthy": True, "provider": self.name, "model": self.settings.model}

    def close(self) -> None:  # pragma: no cover - 默认无资源
        return None


def _raise_for_response(resp: requests.Response, provider: str) -> None:
    if resp.status_code >= 400:
        body = resp.text[:800] if resp.text else ""
        raise LLMError(f"{provider} HTTP {resp.status_code}: {body}")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

class OllamaClient(LLMClient):
    """Ollama ``/api/chat`` 客户端。"""

    def _endpoint(self, suffix: str) -> str:
        base = self.settings.base_url.rstrip("/")
        if base.endswith("/api/chat"):
            base = base[: -len("/api/chat")]
        return f"{base}{suffix}"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.to_wire() for m in messages],
            "stream": False,
            "options": {
                "temperature": self.settings.temperature if temperature is None else temperature,
                "num_predict": self.settings.max_tokens if max_tokens is None else max_tokens,
            },
        }
        if self.settings.num_ctx:
            payload["options"]["num_ctx"] = int(self.settings.num_ctx)
        think = self.settings.thinking if thinking is None else thinking
        if think is not None:
            # Ollama 的 Qwen3/Qwen3.5 使用顶层 think 参数
            payload["think"] = bool(think)
        if self.settings.stop:
            payload["options"]["stop"] = list(self.settings.stop)
        if tools:
            payload["tools"] = list(tools)

        start = time.time()
        try:
            resp = requests.post(
                self._endpoint("/api/chat"),
                json=payload,
                timeout=self.settings.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Ollama 请求失败: {exc}") from exc
        _raise_for_response(resp, "Ollama")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"Ollama 返回非 JSON: {resp.text[:300]}") from exc

        message = data.get("message") or {}
        text = message.get("content") or ""
        calls: List[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if fn.get("name"):
                calls.append(ToolCall(name=str(fn["name"]), arguments=args or {}))

        usage = Usage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
        return LLMResponse(
            text=text.strip(),
            tool_calls=calls,
            usage=usage,
            model=str(data.get("model") or self.settings.model),
            finish_reason=str(data.get("done_reason") or ""),
            raw={"elapsed_s": round(time.time() - start, 3)},
        )

    def health(self) -> Dict[str, Any]:
        start = time.time()
        try:
            resp = requests.get(self._endpoint("/api/tags"), timeout=5)
            _raise_for_response(resp, "Ollama")
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return {
                "healthy": True,
                "provider": "ollama",
                "model": self.settings.model,
                "models": models,
                "model_available": any(self.settings.model.split(":")[0] == m.split(":")[0] for m in models),
                "latency_ms": round((time.time() - start) * 1000, 1),
            }
        except Exception as exc:
            return {
                "healthy": False,
                "provider": "ollama",
                "model": self.settings.model,
                "error": str(exc),
                "latency_ms": round((time.time() - start) * 1000, 1),
            }


# ---------------------------------------------------------------------------
# OpenAI 兼容（OpenAI / vLLM / DeepSeek / 任意兼容网关）
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    """OpenAI Chat Completions 兼容客户端。"""

    def _endpoint(self, suffix: str) -> str:
        base = self.settings.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if base.endswith("/v1"):
            return f"{base}{suffix}"
        # DeepSeek 官方 base_url 不带 /v1（https://api.deepseek.com/chat/completions）
        if self.settings.provider == "deepseek":
            return f"{base}{suffix}"
        return f"{base}/v1{suffix}"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        think = self.settings.thinking if thinking is None else thinking
        if think is not None and (
            self.settings.provider == "deepseek" or "deepseek" in self.settings.base_url.lower()
        ):
            payload["thinking"] = {
                "type": "enabled" if think else "disabled"
            }
        if thinking is not False and self.settings.reasoning_effort and (
            self.settings.provider == "deepseek" or "deepseek" in self.settings.base_url.lower()
        ):
            payload["reasoning_effort"] = self.settings.reasoning_effort
        if self.settings.stop:
            payload["stop"] = list(self.settings.stop)
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"

        try:
            resp = requests.post(
                self._endpoint("/chat/completions"),
                json=payload,
                headers=self._headers(),
                timeout=self.settings.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"OpenAI 兼容接口请求失败: {exc}") from exc
        _raise_for_response(resp, "OpenAI")
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"OpenAI 接口返回非 JSON: {resp.text[:300]}") from exc

        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}
        text = message.get("content") or ""
        calls: List[ToolCall] = []
        for item in message.get("tool_calls") or []:
            fn = item.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"input": args}
            if fn.get("name"):
                calls.append(
                    ToolCall(
                        name=str(fn["name"]),
                        arguments=args or {},
                        id=str(item.get("id") or ""),
                    )
                )
        u = data.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
        )
        return LLMResponse(
            text=text.strip(),
            tool_calls=calls,
            usage=usage,
            model=str(data.get("model") or self.settings.model),
            finish_reason=str(choice.get("finish_reason") or ""),
        )

    def health(self) -> Dict[str, Any]:
        start = time.time()
        try:
            resp = requests.get(
                self._endpoint("/models"),
                headers=self._headers(),
                timeout=5,
            )
            _raise_for_response(resp, "OpenAI")
            models = [m.get("id", "") for m in resp.json().get("data", [])]
            return {
                "healthy": True,
                "provider": self.settings.provider,
                "model": self.settings.model,
                "models": models,
                "model_available": self.settings.model in models if models else None,
                "latency_ms": round((time.time() - start) * 1000, 1),
            }
        except Exception as exc:
            return {
                "healthy": False,
                "provider": self.settings.provider,
                "model": self.settings.model,
                "error": str(exc),
                "latency_ms": round((time.time() - start) * 1000, 1),
            }


# ---------------------------------------------------------------------------
# 脚本化（测试 / 回放）
# ---------------------------------------------------------------------------

ScriptedReply = Union[str, LLMResponse, Callable[[Sequence[Message]], Union[str, LLMResponse]]]


class ScriptedClient(LLMClient):
    """按脚本返回预设响应，用于单元测试与轨迹回放。

    ``repeat_last=True`` 时，脚本耗尽后重复最后一条，适合「一直输出最终答案」的兜底。
    """

    def __init__(self, settings: LLMSettings, replies: Sequence[ScriptedReply], *, repeat_last: bool = False):
        super().__init__(settings)
        self.replies: List[ScriptedReply] = list(replies)
        self.repeat_last = repeat_last
        self.calls: List[List[Message]] = []
        self._last: Optional[ScriptedReply] = None

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        thinking: Optional[bool] = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        if self.replies:
            reply: ScriptedReply = self.replies.pop(0)
            self._last = reply
        elif self.repeat_last and self._last is not None:
            reply = self._last
        else:
            raise LLMError("ScriptedClient 脚本已耗尽")

        if callable(reply):
            reply = reply(messages)
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(text=str(reply), model=self.settings.model, usage=Usage())

    def health(self) -> Dict[str, Any]:
        return {
            "healthy": True,
            "provider": "scripted",
            "model": self.settings.model,
            "remaining": len(self.replies),
        }


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def create_llm(
    settings: Optional[LLMSettings] = None,
    *,
    scripted_replies: Optional[Sequence[ScriptedReply]] = None,
) -> LLMClient:
    """根据配置创建 LLM 客户端。"""
    settings = settings or LLMSettings()
    provider = settings.provider

    if provider == "ollama":
        return OllamaClient(settings)
    if provider in ("openai", "vllm", "deepseek", "openai_compatible"):
        return OpenAIClient(settings)
    if provider == "scripted":
        return ScriptedClient(settings, scripted_replies or [], repeat_last=True)
    if provider in ("offline", "deterministic", "rule"):
        from .offline import OfflinePolicyClient  # 延迟导入避免循环

        return OfflinePolicyClient(settings)
    raise LLMError(f"未知的 LLM provider: {provider!r}")
