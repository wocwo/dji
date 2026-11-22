"""提示词工程：系统提示、工具目录、observation 格式化、预算兜底提示。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import AgentResult, Message, ToolResult


SYSTEM_IDENTITY = """你是「{agent_name}」，一个面向大疆（DJI）无人机与手持设备官方手册的智能 Agent。
你不是一个只会单轮问答的 RAG 机器人：你可以**规划、调用工具、多轮检索、核对证据，再给出结论**。
你的知识来源是本地手册知识库；当证据不足时，你必须明确说明，而不是编造。"""

TOOL_PROTOCOL_TEXT = """## 输出协议（必须严格遵守）

你每次只能输出**一个 JSON 对象**，不要输出 Markdown 代码块围栏，不要输出 JSON 之外的任何文字。

需要调用工具时：
{"thought": "我为什么需要这个工具", "action": "工具名称", "action_input": {"参数名": "参数值"}}

可以给出最终答案时：
{"thought": "为什么证据已经足够", "final_answer": "给用户的完整回答", "citations": ["S1", "S2"]}

规则：
1. 每次只做一个动作；拿到工具结果（observation）后再决定下一步。
2. 只能调用工具目录中存在的工具，参数必须是合法 JSON；不确定参数就用默认值。
3. 工具失败或结果为空时，换关键词/换工具重试；**不要重复完全相同的调用**。
4. `final_answer` 中使用 `[S1]`、`[S2]` 标注依据；引用编号只能来自 observation，且必须紧贴对应句子。
5. 禁止编造手册内容、参数、页码或来源。证据不足时直接说明「本次检索未找到」，不要推断成「手册没有这一章」。
6. 区分「手册明确」「推断」「待确认」；没有机型/版本时不要给具体数值或菜单路径。
7. 不要输出你的内部推理过程以外的客套话，也不要假装调用工具。"""

TOOL_PROTOCOL_NATIVE = """## 工具使用协议

你可以通过原生 function calling 调用工具。需要手册知识时主动调用 `manual_search`；
拿到结果后继续推理或直接回答。回答中使用 `[S1]`、`[S2]` 标注依据，编号只能来自工具结果。
证据不足时明确说明，禁止编造。最终回答使用中文，先结论后细节。"""

FINAL_ANSWER_RULES = """## 回答要求

- 先给直接结论，再给必要步骤/参数；不要复述整个检索过程。
- 引用格式：`[S1]`，紧贴在对应结论/步骤后面；禁止写成 `S1S2S5`，也不要在段尾堆一长串引用。
- 每个事实句都必须能对应到具体片段；引用编号只能来自 observation。
- 明确区分三类内容：
  1. **手册明确**：手册原文直接支持；
  2. **推断**：由手册定义推导，标注「推断：」；
  3. **待确认**：缺少机型/版本/读数，向用户询问。
- 没有机型/版本时，不要给具体数值、菜单路径或“多数情况”这类概括；先问，或明确标注为通用建议。
- 手册未覆盖的内容要说明「本次检索未找到」，不要说「手册中没有这一章」，除非你检索到了目录/索引并能引用。
- 涉及安全操作（电池、桨叶、飞行限制）时保留手册中的警告语义。
- 不要输出「根据我的经验」等手册之外的内容。"""


# ---------------------------------------------------------------------------
# 工具目录
# ---------------------------------------------------------------------------

def _tool_parameters(spec: Any) -> Dict[str, Any]:
    params = getattr(spec, "parameters", None) or {}
    if not isinstance(params, dict):
        return {"type": "object", "properties": {}}
    return params


def render_tool_catalog(tools: Sequence[Any]) -> str:
    """把 ToolSpec 列表渲染为提示词中的工具目录。"""
    if not tools:
        return "（当前没有可用工具）"
    lines: List[str] = []
    for spec in tools:
        name = getattr(spec, "name", "unknown")
        desc = (getattr(spec, "description", "") or "").strip()
        risk = getattr(spec, "risk", "safe")
        params = _tool_parameters(spec)
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []
        lines.append(f"### `{name}`（风险等级：{risk}）")
        lines.append(desc)
        if props:
            lines.append("参数：")
            for pname, pmeta in props.items():
                ptype = pmeta.get("type", "any") if isinstance(pmeta, dict) else "any"
                pdesc = pmeta.get("description", "") if isinstance(pmeta, dict) else ""
                req = "必填" if pname in required else "可选"
                default = pmeta.get("default", None) if isinstance(pmeta, dict) else None
                extra = f"，默认 {default!r}" if default is not None else ""
                lines.append(f"- `{pname}` ({ptype}, {req}{extra})：{pdesc}")
        else:
            lines.append("参数：无")
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 系统提示
# ---------------------------------------------------------------------------

def build_system_prompt(
    tools: Sequence[Any],
    *,
    agent_name: str = "dji-manual-agent",
    kb_description: str = "",
    tool_mode: str = "text",
    max_steps: int = 8,
    now: Optional[datetime] = None,
) -> str:
    """构建系统提示词。"""
    now = now or datetime.now()
    sections: List[str] = [
        SYSTEM_IDENTITY.format(agent_name=agent_name),
        "",
        f"## 环境\n- 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        f"- 本轮最多 {max_steps} 步；请高效使用工具，避免无意义的重复检索。\n"
        f"- 知识库：{kb_description or '大疆官方产品手册（本地向量库 + 词法检索）'}",
        "",
        "## 可用工具",
        render_tool_catalog(tools),
        "",
        TOOL_PROTOCOL_TEXT if tool_mode != "native" else TOOL_PROTOCOL_NATIVE,
        "",
        FINAL_ANSWER_RULES,
    ]
    return "\n".join(sections).strip()


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int) -> str:
    """保持头尾的截断，避免丢掉结果尾部的重要信息。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.72)
    tail = max(0, max_chars - head - 40)
    return f"{text[:head]}\n...（已截断 {len(text) - head - tail} 字符）...\n{text[-tail:] if tail else ''}"


def format_observation(result: ToolResult, *, max_chars: int = 4000) -> str:
    """把工具结果包装成给模型看的 observation 文本。"""
    header = f"[工具结果] {result.tool_name or 'unknown'} | ok={result.ok}"
    if result.latency_ms:
        header += f" | {result.latency_ms:.0f}ms"
    if result.error:
        header += f" | error={result.error}"
    content = result.content or ("（工具没有返回内容）" if result.ok else "（工具执行失败）")
    body = truncate_text(content, max_chars)
    return f"{header}\n{body}"


def format_invalid_action(error: str) -> str:
    """纠正模型输出格式的消息。"""
    return (
        f"你的上一条输出无法执行：{error}\n"
        "请只输出一个 JSON 对象，二选一：\n"
        '{"thought": "...", "action": "工具名", "action_input": {...}}\n'
        '{"thought": "...", "final_answer": "..."}\n'
        "不要输出 Markdown 代码块或额外解释。"
    )


def format_forced_final(reason: str) -> str:
    """预算耗尽时要求模型收尾。"""
    return (
        f"已触发停止条件：{reason}。\n"
        "现在禁止再调用任何工具。请仅基于上面的 observation 给出最终答案；"
        "如果证据不足，就明确说明「现有手册片段不足以回答」，不要编造。\n"
        '只输出：{"thought": "...", "final_answer": "..."}'
    )


def format_evidence_required() -> str:
    """证据闸门拒绝无证据答案时的纠正提示。"""
    return (
        "该问题涉及大疆手册事实，但本轮还没有任何成功的手册检索证据。"
        "请先调用 `manual_search` 检索，再基于 observation 中的 [Sx] 片段作答；"
        "不要凭模型记忆回答，也不要编造页码或参数。"
    )


def format_no_evidence() -> str:
    """已经检索过但没有命中时的收尾提示。"""
    return (
        "本轮已经执行过手册检索，但没有获得可用片段。"
        "请明确说明「根据当前检索到的手册内容，未找到相关信息」，不要编造；"
        "如果缺少机型/版本信息，请向用户澄清。"
    )


def format_missing_facts(missing: Sequence[str]) -> str:
    labels = "、".join(missing)
    return (
        f"你的回答缺少关键事实：{labels}。"
        "请仅基于本轮已检索到的 evidence 做一次定向补充；"
        "如果 evidence 确实没有该信息，明确说明「未找到」，不要编造。"
    )


def format_duplicate_warning(name: str) -> str:
    return (
        f"你刚刚已经执行过 `{name}` 且参数完全相同。请勿重复调用；"
        "要么换关键词/换参数，要么基于已有结果给出最终答案。"
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def trim_history(
    history: Optional[Iterable[Message]],
    *,
    max_chars: int = 8000,
    max_turns: int = 6,
) -> List[Message]:
    """裁剪多轮历史：只保留最近若干条 user/assistant 文本消息。"""
    if not history:
        return []
    messages = [
        m for m in history
        if m.role in ("user", "assistant") and m.content and not m.tool_calls
    ]
    # 以「用户/助手」成对为单位保留
    if max_turns > 0:
        messages = messages[-max_turns * 2:]
    selected: List[Message] = []
    total = 0
    for m in reversed(messages):
        cost = len(m.content)
        if total + cost > max_chars and selected:
            break
        selected.append(m)
        total += cost
    return list(reversed(selected))


def build_result_digest(result: AgentResult, *, max_chars: int = 1200) -> str:
    """把已获得的信息压缩成兜底答案（模型不可用时使用）。"""
    lines: List[str] = []
    for step in result.steps:
        if step.result and step.result.ok and step.result.content:
            lines.append(f"- {step.tool_call.name if step.tool_call else 'tool'}：{step.result.content}")
    digest = "\n".join(lines)
    if not digest:
        return "抱歉，当前未能从手册知识库中获取到足够信息，请补充具体机型或问题细节。"
    return (
        "已达到运行预算，以下是已从手册中检索到的信息（未经过完整归纳）：\n\n"
        + truncate_text(digest, max_chars)
    )


def dumps_action(payload: Dict[str, Any]) -> str:
    """把动作字典序列化为模型可读的 JSON（测试/离线策略用）。"""
    return json.dumps(payload, ensure_ascii=False)
