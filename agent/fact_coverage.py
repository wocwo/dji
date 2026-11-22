"""关键事实覆盖检查：按问题类型定义“必答事实组”。

不是逐字评分，而是检查回答是否覆盖该问题类型的关键事实：
- 电池低温：温度下限、满电、预热、充电限制；
- 充电温度：5/40℃、最佳温度；
- 固件：DJI Fly / DJI Assistant 2；
- 返航：返航点 / 定位；
- 加热：加热方式、时长、目标温度。
缺失时由运行时做一次定向补答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


def _norm(text: str) -> str:
    text = (text or "").lower().replace("°c", "℃")
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", "", text)


@dataclass
class CoverageGroup:
    label: str
    aliases: Sequence[str]

    def hit(self, normalized_answer: str) -> bool:
        return any(_norm(alias) in normalized_answer for alias in self.aliases)


@dataclass
class CoverageReport:
    intent: str
    missing: List[str] = field(default_factory=list)
    checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.missing

    def to_dict(self) -> Dict[str, object]:
        return {"intent": self.intent, "missing": list(self.missing), "checked": self.checked, "passed": self.passed}


_GROUPS: Dict[str, List[CoverageGroup]] = {
    "battery_low_temp": [
        CoverageGroup("温度下限", ["-10"]),
        CoverageGroup("满电", ["满电", "充满"]),
        CoverageGroup("预热", ["预热", "悬停"]),
        CoverageGroup("充电限制", ["5℃", "5°c"]),
    ],
    "battery_charging_temp": [
        CoverageGroup("低温充电下限", ["5℃", "5°c"]),
        CoverageGroup("高温充电上限", ["40℃", "40°c"]),
        CoverageGroup("最佳充电温度", ["25±3", "25℃", "22℃"]),
    ],
    "firmware": [
        CoverageGroup("DJI Fly", ["dji fly"]),
        CoverageGroup("Assistant", ["dji assistant"]),
    ],
    "return": [
        CoverageGroup("返航点", ["返航点"]),
        CoverageGroup("卫星定位", ["卫星定位", "gnss"]),
    ],
    "return_height": [
        CoverageGroup("返航高度", ["返航高度"]),
        CoverageGroup("距离阈值", ["50m", "50 m"]),
        CoverageGroup("当前高度/分档", ["当前高度", "返航距离", "5m"]),
    ],
    "pairing": [
        CoverageGroup("对频", ["对频", "配对"]),
        CoverageGroup("DJI Fly", ["dji fly"]),
    ],
    "compass": [
        CoverageGroup("指南针", ["指南针"]),
        CoverageGroup("校准", ["校准"]),
    ],
    "heat": [
        CoverageGroup("加热", ["加热"]),
        CoverageGroup("启动方式", ["电量开关", "长按", "自动加热"]),
        CoverageGroup("目标温度", ["16", "20℃", "20°c"]),
        CoverageGroup("保温时间", ["30分钟", "30分"]),
    ],
}


class FactCoverageChecker:
    def detect(self, question: str) -> str:
        q = question or ""
        if "对频" in q or "配对" in q:
            return "pairing"
        if "指南针" in q and "校准" in q:
            return "compass"
        if "加热" in q:
            return "heat"
        if "固件" in q:
            return "firmware"
        if "返航" in q and ("高度" in q or "距离" in q):
            return "return_height"
        if "返航" in q:
            return "return"
        if "充电" in q and "温度" in q:
            return "battery_charging_temp"
        if "电池" in q and ("低温" in q or "保养" in q):
            return "battery_low_temp"
        return ""

    def check(self, question: str, answer: str) -> CoverageReport:
        intent = self.detect(question)
        groups = _GROUPS.get(intent, [])
        if not groups:
            return CoverageReport(intent=intent)
        normalized = _norm(answer)
        missing = [group.label for group in groups if not group.hit(normalized)]
        return CoverageReport(intent=intent, missing=missing, checked=len(groups))


__all__ = ["FactCoverageChecker", "CoverageReport", "CoverageGroup"]
