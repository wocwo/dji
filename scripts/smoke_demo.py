"""最小 HTTP 冒烟测试：健康检查 + 一次带引用的问答。

服务需先启动：
    python -m agent serve --port 8077 --config configs/agent.deepseek.yaml
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def auth_headers(user: str, password: str) -> dict:
    raw = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {raw}"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8077")
    parser.add_argument("--user", default=os.environ.get("AGENT_USER", "dji"))
    parser.add_argument("--password", default=os.environ.get("AGENT_ACCESS_TOKEN", ""))
    parser.add_argument("--question", default="Mini 4 Pro 电池低温怎么保养？")
    args = parser.parse_args()

    headers = auth_headers(args.user, args.password) if args.password else {}
    health = requests.get(f"{args.base_url}/healthz", headers=headers, timeout=20)
    health.raise_for_status()
    payload = health.json()
    print("health:", payload.get("agent"), "llm:", payload.get("llm", {}).get("healthy"), "kb:", payload.get("knowledge", {}).get("count"))

    created = requests.post(
        f"{args.base_url}/v1/agent/jobs",
        json={"task": args.question, "mode": "auto"},
        headers=headers,
        timeout=20,
    )
    created.raise_for_status()
    job_id = created.json()["job_id"]
    started = time.time()
    for _ in range(120):
        response = requests.get(f"{args.base_url}/v1/agent/jobs/{job_id}", headers=headers, timeout=30)
        response.raise_for_status()
        job = response.json()
        if job.get("status") in ("done", "error"):
            break
        time.sleep(0.8)
    else:
        raise SystemExit("job timeout")

    if job.get("status") != "done":
        raise SystemExit(f"job failed: {job}")
    result = job.get("result") or {}
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    print(f"status: done · {round(time.time() - started, 1)}s · citations: {len(citations)}")
    print(answer[:240].replace("\n", " "))
    if not answer or not citations:
        raise SystemExit("smoke failed: empty answer or no citations")
    print(json.dumps({"citations": citations[:3]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
