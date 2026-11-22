"""Qwen-VL 视觉解析测试（不访问网络）。"""

from __future__ import annotations

from agent.vision import QwenVLClient, VisionObservation


def test_parse_observation():
    obs = QwenVLClient._parse(
        '{"model_guess":"DJI Mini 4 Pro","screen_text":"Battery Low",'
        '"leds":"LED1 闪烁","visible_damage":{"battery":"疑似鼓包"},'
        '"confidence":0.82,"uncertain":["云台排线"]}'
    )
    assert obs.available
    assert obs.model_guess == "DJI Mini 4 Pro"
    assert obs.visible_damage["battery"] == "疑似鼓包"
    assert obs.confidence == 0.82
    assert "视觉观察" in obs.to_context()


def test_no_key_falls_back():
    obs = QwenVLClient(api_key="").analyze(b"", question="这是什么")
    assert not obs.available
    assert "DASHSCOPE_API_KEY" in obs.error


def test_markdown_json_is_parsed():
    obs = QwenVLClient._parse('```json\n{"model_guess":"Mini 4 Pro","confidence":0.5}\n```')
    assert obs.model_guess == "Mini 4 Pro"
    assert obs.confidence == 0.5


def test_native_maas_mode():
    client = QwenVLClient(
        api_key="sk-test",
        base_url="https://cn-beijing.maas.qianwenaiapi.com/api/v1",
        model="qwen-vl-ocr",
    )
    assert client.mode == "native"
    assert client.model == "qwen-vl-ocr"
