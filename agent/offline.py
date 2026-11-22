"""离线确定性策略：不依赖任何 LLM 也能跑通完整 Agent 循环。

它不是「假 Agent」：同样经过 规划 → 工具调用 → observation → 再决策 → 最终答案，
只是把“模型决策”换成了可预测的规则。用途：
- CI / 单元测试不依赖 Ollama；
- 没有 GPU 或网络时演示完整链路；
- 作为真实 LLM 不可用时的降级方案。

策略：
1. 算术题 → calculator；
2. 其他问题 → manual_search；
3. 无结果时用简化关键词再搜一次；
4. 从 observation 的 [Sx] 片段中抽取句子，生成带引用的答案。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from .config import LLMSettings
from .llm import LLMClient, LLMResponse
from .prompt import truncate_text
from .types import Message, Usage


_BLOCK_RE = re.compile(
    r"^\[(?P<ref>S\d+)\]\s*(?P<header>.*?)\n(?P<body>.*?)(?=^\[S\d+\]|\Z)",
    re.MULTILINE | re.DOTALL,
)
_ARITHMETIC_RE = re.compile(r"^[\d\s\+\-\*/%\(\)\.\^×÷]+$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|\n+")
_SIMPLIFY_RE = re.compile(
    r"(请问|帮我|我想知道|我想问|想知道|告诉我|介绍一下|说明一下|怎么|如何|为什么|为何|"
    r"什么|哪些|是否|可以|能否|需要|应该|一下|吗|呢|吧|啊|呀|的|了)"
)
_TABLE_LINE_RE = re.compile(r"^\s*[\|\-\:\s0-9]+\s*$")
_NOISE_MARKERS = (
    "版权所有",
    "©",
    "LED1",
    "LED2",
    "LED3",
    "LED4",
    "Samsung EVO",
    "microSDXC",
    "mAh",
    "Wh",
)


def _extract_task(messages: Sequence[Message]) -> str:
    """取最后一条真正的用户任务（跳过工具 observation 与收尾提示）。"""
    for message in reversed(messages):
        if message.role != "user":
            continue
        content = message.content.strip()
        if not content:
            continue
        if content.startswith("[工具结果]"):
            continue
        if content.startswith("已触发停止条件"):
            continue
        if content.startswith("你的上一条输出无法执行"):
            continue
        return content
    return ""


def _observations(messages: Sequence[Message]) -> List[str]:
    return [
        m.content
        for m in messages
        if m.role == "user" and m.content.startswith("[工具结果]")
    ]


def _parse_passages(observation: str) -> List[Dict[str, str]]:
    passages: List[Dict[str, str]] = []
    for match in _BLOCK_RE.finditer(observation):
        body = match.group("body").strip()
        if not body:
            continue
        passages.append(
            {
                "ref": match.group("ref").strip(),
                "header": match.group("header").strip(),
                "body": body,
            }
        )
    return passages


def _simplify(query: str) -> str:
    text = _SIMPLIFY_RE.sub(" ", query)
    text = re.sub(r"[，。？！、,.?!:：；;\"'“”‘’]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _pick_sentences(body: str, query: str, limit: int = 2) -> List[str]:
    """从片段中挑选与问题最相关的句子（而不是机械地取前两句）。"""
    from .knowledge import extract_terms, tokenize

    cleaned_lines: List[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or _TABLE_LINE_RE.match(line):
            continue
        cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    focus_terms = [
        term for term in extract_terms(query, max_terms=8)
        if term in ("低温", "高温", "固件", "对频", "避障", "返航", "校准", "充电", "存储")
    ]
    sentences: List[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        part = part.strip(" -\t•◆●▪·")
        if len(part) < 8:
            continue
        if any(marker in part for marker in _NOISE_MARKERS):
            continue
        # 问题带有“低温/固件”等强主题词时，过滤掉同一页里的通用说明
        if focus_terms and not any(term in part for term in focus_terms):
            continue
        # 纯参数表/指示灯表格行：数字占比过高且没有操作建议
        digits = sum(char.isdigit() for char in part)
        if digits / max(len(part), 1) > 0.28 and not any(
            keyword in part for keyword in ("建议", "请", "禁止", "务必", "注意", "需")
        ):
            continue
        if len(part) > 220:
            part = part[:220] + "…"
        sentences.append(part)

    query_terms = {token for token in tokenize(query) if len(token) >= 2}
    scored: List[tuple] = []
    for index, sentence in enumerate(sentences):
        hits = sum(1 for term in query_terms if term in sentence.lower())
        actionable = any(
            keyword in sentence
            for keyword in ("建议", "请", "禁止", "务必", "注意", "无法", "需", "先", "切勿")
        )
        score = hits * 2.0 + (0.8 if actionable else 0.0) + min(len(sentence), 120) / 240
        scored.append((score, index, sentence))
    if not scored:
        return []
    chosen = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    chosen.sort(key=lambda item: item[1])
    return [sentence for _, _, sentence in chosen]


def _compose_answer(passages: Sequence[Dict[str, str]], query: str = "") -> str:
    lines: List[str] = ["根据大疆官方手册片段：", ""]
    used = 0
    for index, passage in enumerate(passages[:3]):
        # 首条证据多展示几句，其余保持精简，避免离线答案变成片段堆砌
        sentences = _pick_sentences(
            passage["body"],
            query,
            limit=3 if index == 0 else 1,
        )
        if not sentences:
            continue
        used += 1
        for sentence in sentences:
            lines.append(f"- {sentence} [{passage['ref']}]")
    if not used:
        first = passages[0]
        snippet = truncate_text(re.sub(r"\s+", " ", first["body"]).strip(), 240)
        lines.append(f"- {snippet} [{first['ref']}]")
    lines += [
        "",
        "以上信息来自手册检索片段，具体操作与参数请以对应机型的最新官方手册为准。",
    ]
    return "\n".join(lines)


class OfflinePolicyClient(LLMClient):
    """规则驱动的确定性“模型”，用于离线演示与测试。"""

    def __init__(self, settings: Optional[LLMSettings] = None):
        super().__init__(settings or LLMSettings(provider="offline", model="offline-policy"))

    @property
    def name(self) -> str:
        return "offline"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        task = _extract_task(messages)
        observations = _observations(messages)

        # 0) 上一条是重复调用警告 → 不要再搜
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "已经执行过" in last_user or "重复" in last_user:
            passages: List[Dict[str, str]] = []
            for obs in reversed(observations):
                passages = _parse_passages(obs)
                if passages:
                    break
            answer = (
                _compose_answer(passages, task)
                if passages
                else "现有手册片段不足，未找到可确认的答案。请补充具体机型或问题细节。"
            )
            return self._final(answer)

        # 1) 算术题：支持“请计算 (5000-249)/1000”这类自然语言包裹
        compact = task.replace(" ", "")
        calc_words = ("计算", "等于", "乘以", "除以", "加上", "减去", "换算")
        math_part = re.sub(r"[^\d\+\-\*/%\(\)\.\^×÷\s]", " ", task).strip()
        has_operator = bool(re.search(r"[\+\-\*/%×÷\^]", math_part))
        has_number = bool(re.search(r"\d", math_part))
        if (
            (_ARITHMETIC_RE.fullmatch(compact) or any(word in task for word in calc_words) or "等于" in task)
            and has_operator
            and has_number
        ):
            if not any("calculator" in obs for obs in observations):
                expression = re.sub(r"\s+", " ", math_part).strip()
                return self._tool("calculator", {"expression": expression})
            for obs in reversed(observations):
                if "calculator" in obs:
                    # 避免命中 observation 头部 "ok=True"，只取计算结果行
                    match = re.search(r"计算结果[:：].*?=\s*([^\n]+)", obs)
                    result = match.group(1).strip() if match else ""
                    return self._final(f"计算结果是 {result}。")

        # 2) 第一次检索
        search_obs = [obs for obs in observations if obs.startswith("[工具结果] manual_search")]
        if not search_obs:
            return self._tool(
                "manual_search",
                {"query": task or "大疆 手册", "top_k": 4},
            )

        # 3) 检索为空 / 失败 → 换简化关键词再试一次
        last_obs = search_obs[-1]
        passages = _parse_passages(last_obs)
        first_line = last_obs.splitlines()[0] if last_obs else ""
        failed = "ok=False" in first_line or "未检索到" in last_obs
        if (failed or not passages) and len(search_obs) == 1:
            simplified = _simplify(task)
            if simplified and simplified.lower() != task.lower():
                return self._tool(
                    "manual_search",
                    {"query": simplified, "top_k": 4},
                )

        # 4) 汇总作答
        if not passages:
            return self._final(
                "根据当前检索到的大疆官方手册内容，未找到与该问题直接相关的信息。"
                "建议补充具体机型，或查阅对应机型的最新用户手册。"
            )
        return self._final(_compose_answer(passages, task))

    # -- 输出构造 ---------------------------------------------------------
    @staticmethod
    def _tool(name: str, arguments: Dict[str, Any]) -> LLMResponse:
        payload = {
            "thought": f"需要调用 {name} 获取依据。",
            "action": name,
            "action_input": arguments,
        }
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), usage=Usage(), model="offline-policy")

    @staticmethod
    def _final(answer: str) -> LLMResponse:
        payload = {
            "thought": "已有足够的手册片段，可以基于引用作答。",
            "final_answer": answer,
        }
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), usage=Usage(), model="offline-policy")

    def health(self) -> Dict[str, Any]:
        return {"healthy": True, "provider": "offline", "model": "offline-policy"}


__all__ = ["OfflinePolicyClient"]
