"""动作解析器测试：覆盖模型输出的各种“不守规矩”形态。"""

from __future__ import annotations

import json

import pytest

from agent.parser import count_json_objects, extract_citation_refs, parse_action
from tests.helpers import action, fenced, final


def test_parse_tool_json():
    parsed = parse_action(action("manual_search", query="电池 低温"))
    assert parsed.kind == "tool"
    assert parsed.tool_call is not None
    assert parsed.tool_call.name == "manual_search"
    assert parsed.tool_call.arguments == {"query": "电池 低温"}
    assert "manual_search" in parsed.thought


def test_parse_final_json():
    payload = json.dumps(
        {"thought": "证据足够", "final_answer": "应先预热电池。[S1]", "citations": ["S1"]},
        ensure_ascii=False,
    )
    parsed = parse_action(payload)
    assert parsed.kind == "final"
    assert parsed.final_answer == "应先预热电池。[S1]"
    assert parsed.thought == "证据足够"


def test_parse_fenced_json_with_prose():
    text = "好的，我来检索：\n" + fenced(action("manual_search", query="固件")) + "\n请稍等。"
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.arguments["query"] == "固件"


def test_parse_trailing_comma():
    text = '{"action": "calculator", "action_input": {"expression": "1+1",}}'
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.name == "calculator"


def test_parse_action_input_as_json_string():
    text = json.dumps(
        {"action": "manual_search", "action_input": '{"query": "遥控器 对频"}'},
        ensure_ascii=False,
    )
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.arguments == {"query": "遥控器 对频"}


def test_parse_action_object_form():
    text = json.dumps(
        {"action": {"name": "calculator", "arguments": {"expression": "2+2"}}},
        ensure_ascii=False,
    )
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.name == "calculator"
    assert parsed.tool_call.arguments == {"expression": "2+2"}


def test_parse_braces_inside_thought():
    text = json.dumps(
        {"thought": "比较 {A} 与 {B} 后决定", "action": "clock", "action_input": {}},
        ensure_ascii=False,
    )
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.name == "clock"


def test_parse_react_style():
    text = (
        "Thought: 我需要查手册\n"
        "Action: manual_search\n"
        'Action Input: {"query": "智能飞行电池"}\n'
    )
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.name == "manual_search"
    assert parsed.tool_call.arguments == {"query": "智能飞行电池"}


def test_parse_react_final_answer():
    text = "Thought: 可以回答\nFinal Answer: 电池应存放在 40%-60% 电量。"
    parsed = parse_action(text)
    assert parsed.kind == "final"
    assert "40%-60%" in parsed.final_answer


def test_plain_text_becomes_soft_final():
    parsed = parse_action("智能飞行电池应避免在低温下充电。")
    assert parsed.kind == "final"
    assert parsed.soft is True
    assert "低温" in parsed.final_answer


def test_plain_text_rejected_when_disabled():
    parsed = parse_action("随便说点什么", plain_text_final=False)
    assert parsed.kind == "invalid"
    assert parsed.error


def test_empty_output_invalid():
    parsed = parse_action("   ")
    assert parsed.kind == "invalid"
    assert "为空" in parsed.error


def test_json_without_action_or_final_invalid():
    parsed = parse_action('{"thought": "只有思考"}', plain_text_final=False)
    assert parsed.kind == "invalid"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("结论 [S1]。", ["S1"]),
        ("[S1] 和 [S2]", ["S1", "S2"]),
        ("[S1，S3] 与 [S2, S3]", ["S1", "S3", "S2"]),
        ("［S1][S2］ 全角括号", ["S1", "S2"]),
        ("[S1、S2]", ["S1", "S2"]),
        ("没有引用", []),
        ("[s1] 小写", ["S1"]),
    ],
)
def test_extract_citation_refs(answer, expected):
    assert extract_citation_refs(answer) == expected


def test_multiple_json_objects_picks_action():
    text = '{"foo": 1}\n{"action": "clock", "action_input": {}}'
    parsed = parse_action(text)
    assert parsed.kind == "tool"
    assert parsed.tool_call.name == "clock"


def test_count_json_objects():
    assert count_json_objects('{"a":1} {"b":2}') >= 2
