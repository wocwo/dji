"""测试辅助函数。"""

from __future__ import annotations

import json
from typing import Any, Dict


def action(name: str, **arguments: Any) -> str:
    """构造文本协议的工具调用。"""
    return json.dumps(
        {
            "thought": f"需要调用 {name}",
            "action": name,
            "action_input": arguments,
        },
        ensure_ascii=False,
    )


def final(answer: str, citations: Any = None) -> str:
    """构造文本协议的最终答案。"""
    payload: Dict[str, Any] = {"thought": "可以作答", "final_answer": answer}
    if citations is not None:
        payload["citations"] = citations
    return json.dumps(payload, ensure_ascii=False)


def fenced(payload: str) -> str:
    return f"```json\n{payload}\n```"
