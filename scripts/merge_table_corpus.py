"""把表格记录并入页面语料。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass


def _read(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("text"):
                items.append(item)
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="合并页面与表格语料")
    parser.add_argument("--pages", default="data/agent_manual_pages.jsonl")
    parser.add_argument("--tables", default="data/agent_manual_tables.jsonl")
    parser.add_argument("--output", default="data/agent_manual_kb.jsonl")
    args = parser.parse_args()

    pages = _read(Path(args.pages))
    tables = _read(Path(args.tables))
    merged: Dict[str, Dict[str, Any]] = {}
    for item in pages + tables:
        key = str(item.get("chunk_id") or f"{item.get('doc_id')}:{item.get('page')}:{item.get('text','')[:30]}")
        merged[key] = item
    records = list(merged.values())
    records.sort(
        key=lambda item: (
            str(item.get("model") or ""),
            str(item.get("doc_id") or ""),
            int(item.get("page") or 0) if str(item.get("page") or "0").isdigit() else 0,
            str(item.get("type") or ""),
            str(item.get("chunk_id") or ""),
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)

    types = Counter(str(item.get("type") or "") for item in records)
    print(
        json.dumps(
            {
                "pages": len(pages),
                "tables": len(tables),
                "records": len(records),
                "types": types.most_common(),
                "chars": sum(len(item.get("text") or "") for item in records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"写出: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
