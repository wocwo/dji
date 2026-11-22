"""关键事实覆盖检查测试。"""

from __future__ import annotations

from agent.fact_coverage import FactCoverageChecker


def test_battery_low_temp_missing():
    report = FactCoverageChecker().check("Mavic 4 Pro 电池低温怎么保养？", "电池要满电。")
    assert report.intent == "battery_low_temp"
    assert "温度下限" in report.missing
    assert "预热" in report.missing


def test_battery_low_temp_passes():
    answer = "低温下 -10℃ 以下不能飞；要保持满电并预热；5℃ 以下不要充电。"
    report = FactCoverageChecker().check("Mavic 4 Pro 电池低温怎么保养？", answer)
    assert report.passed


def test_heat_requires_duration():
    report = FactCoverageChecker().check(
        "Matrice 350 RTK 电池怎么手动加热？",
        "关机状态下长按电量开关 4 秒，温度保持在 16-20℃。",
    )
    assert "保温时间" in report.missing


def test_return_height_requires_threshold():
    report = FactCoverageChecker().check(
        "Neo 2 的返航距离和高度关系是什么？",
        "返航点依赖卫星定位，定位良好才能返航。",
    )
    assert report.intent == "return_height"
    assert "返航高度" in report.missing
    assert "距离阈值" in report.missing


def test_unknown_intent_skips():
    report = FactCoverageChecker().check("你好", "你好")
    assert report.passed
    assert report.checked == 0
