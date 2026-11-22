"""本地 ONNX cross-encoder reranker（bge-reranker-base）。

不依赖 torch/transformers：
- tokenizers 读取 tokenizer.json
- onnxruntime 做 CPU 推理

默认模型路径：``.agent/models/bge-reranker-base``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_dir: str | Path = ".agent/models/bge-reranker-base", *, max_length: int = 256):
        self.model_dir = Path(model_dir)
        self.max_length = int(os.environ.get("AGENT_RERANK_MAX_LENGTH", max_length))
        self._tokenizer = None
        self._session = None
        self._input_names: set[str] = set()

    @property
    def available(self) -> bool:
        return (self.model_dir / "onnx" / "model.onnx").exists() and (self.model_dir / "tokenizer.json").exists()

    def _load(self) -> None:
        if self._session is not None:
            return
        from tokenizers import Tokenizer
        import onnxruntime as ort

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding()
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(self.model_dir / "onnx" / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    def rank(
        self,
        query: str,
        candidates: Sequence[Tuple[float, Any]],
        *,
        top_k: int,
        text_chars: int = 700,
        base_weight: float = 0.5,
    ) -> List[Tuple[float, Any]]:
        base_weight = float(os.environ.get("AGENT_RERANK_BASE_WEIGHT", base_weight))
        text_chars = int(os.environ.get("AGENT_RERANK_TEXT_CHARS", text_chars))
        if not candidates:
            return []
        if not self.available:
            return list(candidates[:top_k])
        self._load()
        pairs = []
        for _base, passage in candidates:
            title = str(getattr(passage, "title", "") or "")
            section = str(getattr(passage, "section", "") or "")
            body = str(getattr(passage, "text", "") or "")[:text_chars]
            pairs.append((query, f"{title} {section} {body}".strip()))
        encodings = self._tokenizer.encode_batch(pairs)
        feed = {
            "input_ids": np.array([item.ids for item in encodings], dtype=np.int64),
            "attention_mask": np.array([item.attention_mask for item in encodings], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([item.type_ids for item in encodings], dtype=np.int64)
        outputs = self._session.run(None, {key: value for key, value in feed.items() if key in self._input_names})
        cross_scores = outputs[0].reshape(-1).tolist()
        base_scores = [float(base) for base, _ in candidates]
        base_max = max(base_scores) if base_scores else 1.0
        base_max = base_max or 1.0
        cross_min = min(cross_scores) if cross_scores else 0.0
        cross_max = max(cross_scores) if cross_scores else 1.0
        cross_span = (cross_max - cross_min) or 1.0
        ranked = []
        for base, cross, (_old, passage) in zip(base_scores, cross_scores, candidates):
            base_norm = max(0.0, min(1.0, base / base_max))
            cross_norm = (cross - cross_min) / cross_span
            score = base_weight * base_norm + (1.0 - base_weight) * cross_norm
            ranked.append((score, passage))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[: max(1, int(top_k))]


def default_reranker() -> CrossEncoderReranker | None:
    path = os.environ.get("AGENT_RERANK_MODEL")
    if not path:
        return None
    reranker = CrossEncoderReranker(path)
    return reranker if reranker.available else None


__all__ = ["CrossEncoderReranker", "default_reranker"]
