"""视觉评测：可复现的错误截图 + 公开产品图 + 空白图。

用法：
    python scripts/eval_vision.py --output reports/vision_eval_2026-11-22.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_LIBS = ROOT / ".agent" / "ocrlibs"
if _LIBS.exists() and str(_LIBS) not in sys.path:
    sys.path.insert(0, str(_LIBS))

from PIL import Image, ImageDraw  # noqa: E402

from agent.vision import QwenVLClient  # noqa: E402
from agent.config import load_env_file  # noqa: E402


def _generated_screenshot() -> bytes:
    image = Image.new("RGB", (640, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 45), "DJI Fly ERROR: Battery Low", fill="black")
    draw.text((20, 100), "Please charge the battery", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _blank() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def run(cases_path: str, output: str) -> int:
    load_env_file()
    client = QwenVLClient()
    rows = []
    for line in Path(cases_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        kind = case["kind"]
        if kind == "generated":
            data = _generated_screenshot()
        elif kind == "blank":
            data = _blank()
        else:
            response = requests.get(case["url"], timeout=40, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            data = response.content
        started = time.time()
        obs = client.analyze(data, question=case.get("question", ""))
        checks = {
            "available": not obs.error,
            "screen": case.get("expect_screen_contains", "").lower() in obs.screen_text.lower()
            if case.get("expect_screen_contains") else True,
            "model": case.get("expect_model_contains", "").lower() in obs.model_guess.lower()
            if case.get("expect_model_contains") else True,
            "empty_damage": not obs.visible_damage if case.get("expect_empty_damage") else True,
            "no_damage": not obs.visible_damage if case.get("expect_no_hallucinated_damage") else True,
        }
        row = {
            "id": case["id"],
            "passed": all(checks.values()),
            "checks": checks,
            "observation": obs.to_dict(),
            "duration_ms": round((time.time() - started) * 1000, 1),
            "tags": case.get("tags", []),
        }
        rows.append(row)
        print(case["id"], "PASS" if row["passed"] else "FAIL", row["duration_ms"], checks, flush=True)
    report = {
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "rows": rows,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{report['passed']}/{report['total']} passed · {output}")
    return 0 if report["passed"] == report["total"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/vision_eval.jsonl")
    parser.add_argument("--output", default="reports/vision_eval.json")
    args = parser.parse_args()
    return run(args.cases, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
