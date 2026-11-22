"""Qwen-VL 云端视觉观察器（DashScope OpenAI-compatible）。

只输出结构化观察，不直接给维修结论。
没有 API Key 时返回不可用，由上层回退到 OCR。
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VisionObservation:
    model_guess: str = ""
    screen_text: str = ""
    leds: str = ""
    visible_damage: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    uncertain: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def available(self) -> bool:
        return not self.error and bool(
            self.screen_text or self.leds or self.visible_damage or self.model_guess
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_guess": self.model_guess,
            "screen_text": self.screen_text,
            "leds": self.leds,
            "visible_damage": dict(self.visible_damage),
            "confidence": round(float(self.confidence), 3),
            "uncertain": list(self.uncertain),
            "error": self.error,
        }

    def to_context(self) -> str:
        if self.error:
            return ""
        parts = ["视觉观察（非手册，仅描述可见内容）"]
        if self.model_guess:
            parts.append(f"- 机型猜测：{self.model_guess}")
        if self.screen_text:
            parts.append(f"- 屏幕/报错文字：{self.screen_text}")
        if self.leds:
            parts.append(f"- 指示灯：{self.leds}")
        if self.visible_damage:
            parts.append("- 可见损伤：")
            for key, value in self.visible_damage.items():
                if value:
                    parts.append(f"  - {key}：{value}")
        if self.uncertain:
            parts.append("- 不确定：" + "；".join(self.uncertain))
        parts.append(f"- 视觉置信度：{self.confidence:.0%}")
        return "\n".join(parts)


class QwenVLClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        if api_key is not None:
            self.api_key = str(api_key).strip()
        else:
            self.api_key = (os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or "").strip()
        self.model = model or os.environ.get("QWEN_VL_MODEL", "qwen-vl-max")
        self.ocr_model = os.environ.get("QWEN_OCR_MODEL", "qwen-vl-ocr")
        self.base_url = (
            base_url
            or os.environ.get("QWEN_BASE_URL")
            or "https://cn-beijing.maas.qianwenaiapi.com/api/v1"
        ).rstrip("/")
        self.mode = (os.environ.get("QWEN_MODE") or ("native" if ("qianwenaiapi" in self.base_url or "maas" in self.base_url) else "openai")).lower()
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _encode(self, image_bytes: bytes, mime: str) -> tuple[str, str]:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.thumbnail((1280, 1280))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=82, optimize=True)
        data = output.getvalue()
        return "image/jpeg", "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

    def _prompt(self, question: str, *, ocr_only: bool = False) -> str:
        if ocr_only:
            return (
                "请仅输出图像中的文本内容。不要解释，不要 Markdown，不要补充。"
                + (f"\n用户问题：{question[:300]}" if question else "")
            )
        return (
            "你是无人机事故图片的视觉观察器，不是维修工程师。"
            "只描述图片中能看见的内容，不推断内部损坏，不给维修结论，不编造。"
            "只输出一个 JSON 对象，不要 Markdown，不要解释。字段：\n"
            '{"model_guess":"机型或空","screen_text":"屏幕/报错文字","leds":"指示灯状态",'
            '"visible_damage":{"arms":"","gimbal":"","battery":"","body":""},'
            '"confidence":0.0,"uncertain":["..."]}\n'
            "confidence 是 0-1；看不清就降低 confidence 并写入 uncertain。"
            + (f"\n用户问题：{question[:500]}" if question else "")
        )

    def _native_analyze(self, image_bytes: bytes, mime: str, question: str, *, prefer_ocr: bool = False) -> VisionObservation:
        """DashScope 原生 MultiModalConversation，支持用户提供的 MaaS 网关。"""
        try:
            import dashscope
            from dashscope import MultiModalConversation

            dashscope.base_http_api_url = self.base_url
            _jpeg, data_url = self._encode(image_bytes, mime)
            ocr_only = prefer_ocr or "ocr" in self.model.lower()
            model = self.ocr_model if prefer_ocr else self.model
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"image": data_url},
                            {"text": self._prompt(question, ocr_only=ocr_only)},
                        ],
                    }
                ],
            )
            status = int(getattr(response, "status_code", 200) or 200)
            if status >= 400:
                message = getattr(response, "message", "") or str(response)
                return VisionObservation(error=f"Qwen-VL HTTP {status}: {message[:180]}")
            content = response.output.choices[0].message.content
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) for item in content if isinstance(item, dict)
                )
            text = str(content).strip()
            if ocr_only:
                return VisionObservation(screen_text=text, confidence=0.9)
            return self._parse(text)
        except Exception as exc:
            return VisionObservation(error=f"{type(exc).__name__}: {str(exc)[:180]}")

    def analyze(self, image_bytes: bytes, *, mime: str = "image/png", question: str = "", prefer_ocr: bool = False) -> VisionObservation:
        if not self.available:
            return VisionObservation(error="未配置 DASHSCOPE_API_KEY / QWEN_API_KEY")
        if self.mode == "native":
            return self._native_analyze(image_bytes, mime, question, prefer_ocr=prefer_ocr)
        try:
            import requests

            _mime, data_url = self._encode(image_bytes, mime)
            prompt = (
                "你是无人机事故图片的视觉观察器，不是维修工程师。"
                "只描述图片中能看见的内容，不推断内部损坏，不给维修结论，不编造。"
                "只输出一个 JSON 对象，不要 Markdown，不要解释。字段：\n"
                '{"model_guess":"机型或空","screen_text":"屏幕/报错文字","leds":"指示灯状态",'
                '"visible_damage":{"arms":"","gimbal":"","battery":"","body":""},'
                '"confidence":0.0,"uncertain":["..."]}\n'
                "confidence 是 0-1；看不清就降低 confidence 并写入 uncertain。"
                + (f"\n用户问题：{question[:500]}" if question else "")
            )
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "temperature": 0.0,
                    "max_tokens": 800,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                },
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                return VisionObservation(error=f"Qwen-VL HTTP {response.status_code}: {response.text[:160]}")
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            return self._parse(str(content))
        except Exception as exc:
            return VisionObservation(error=f"{type(exc).__name__}: {str(exc)[:160]}")

    @staticmethod
    def _clean_text(value: Any) -> str:
        text = str(value or "").strip()
        if text.lower() in {"空", "无", "none", "null", "未知", "n/a", "na"}:
            return ""
        return text

    @staticmethod
    def _parse(text: str) -> VisionObservation:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            cleaned = match.group(0)
        try:
            data = json.loads(cleaned)
        except Exception:
            return VisionObservation(error="Qwen-VL 返回内容不是 JSON", screen_text=text.strip()[:300])
        damage = data.get("visible_damage") or {}
        if not isinstance(damage, dict):
            damage = {"notes": str(damage)}
        try:
            confidence = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        uncertain = data.get("uncertain") or []
        if isinstance(uncertain, str):
            uncertain = [uncertain]
        clean_damage = {}
        for key, value in damage.items():
            cleaned = QwenVLClient._clean_text(value)
            if cleaned:
                clean_damage[str(key)] = cleaned
        return VisionObservation(
            model_guess=QwenVLClient._clean_text(data.get("model_guess")),
            screen_text=QwenVLClient._clean_text(data.get("screen_text")),
            leds=QwenVLClient._clean_text(data.get("leds")),
            visible_damage=clean_damage,
            confidence=max(0.0, min(1.0, confidence)),
            uncertain=[QwenVLClient._clean_text(item) for item in uncertain if QwenVLClient._clean_text(item)],
        )


__all__ = ["QwenVLClient", "VisionObservation"]
