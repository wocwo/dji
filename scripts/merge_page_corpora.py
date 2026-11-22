"""合并 PDF 精读页面与旧 chunk 页面兜底语料。

PDF 精读覆盖核心机型（用户手册/安全指引/产品信息），页码和正文更可靠；
其余文档继续使用按页合并的旧 chunk 语料。PDF 已覆盖的 doc_id 整本替换，
避免旧语料里的 page=0 重复记录混进来。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", urllib.parse.unquote(str(value)).lower())


def _read(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("text"):
                items.append(item)
    return items


def merge(pdf_core: Any, chunk_pages: Path, output: Path) -> Dict[str, Any]:
    pdf_paths = list(pdf_core) if isinstance(pdf_core, (list, tuple)) else [pdf_core]
    pdf_items: List[Dict[str, Any]] = []
    for path in pdf_paths:
        pdf_items.extend(_read(Path(path)))
    chunk_items = _read(chunk_pages)
    pdf_docs = {_compact(item.get("doc_id", "")) for item in pdf_items if item.get("doc_id")}

    merged: Dict[str, Dict[str, Any]] = {}
    for item in chunk_items:
        if _compact(item.get("doc_id", "")) in pdf_docs:
            continue
        item.setdefault("source_kind", "chunk")
        key = str(item.get("chunk_id") or f"{item.get('doc_id')}:{item.get('page')}:{item.get('text','')[:40]}")
        merged[key] = item
    for item in pdf_items:
        item.setdefault("source_kind", "pdf")
        key = str(item.get("chunk_id") or f"{item.get('doc_id')}:{item.get('page')}:{item.get('text','')[:40]}")
        merged[key] = item

    records = list(merged.values())
    records.sort(
        key=lambda item: (
            str(item.get("model") or ""),
            str(item.get("doc_id") or ""),
            int(item.get("page") or 0) if str(item.get("page") or "0").isdigit() else 0,
            str(item.get("chunk_id") or ""),
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp.replace(output)

    models = Counter(str(item.get("model") or "") for item in records)
    types = Counter(str(item.get("doc_type") or "") for item in records)
    docs = len({_compact(item.get("doc_id", "")) for item in records})
    return {
        "pdf_records": len(pdf_items),
        "chunk_records": len(chunk_items),
        "pdf_docs": len(pdf_docs),
        "records": len(records),
        "docs": docs,
        "chars": sum(len(item.get("text") or "") for item in records),
        "models": models.most_common(30),
        "types": types.most_common(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="合并页面语料")
    parser.add_argument(
        "--pdf-core",
        action="append",
        default=None,
        help="PDF 精读页面文件（可重复）",
    )
    parser.add_argument("--chunk-pages", default="data/agent_manual_pages_from_chunks.jsonl")
    parser.add_argument("--output", default="data/agent_manual_pages.jsonl")
    args = parser.parse_args()
    pdf_paths = args.pdf_core or [
        "data/agent_manual_pages_pdf_core.jsonl",
        "data/agent_manual_pages_pdf_extra.jsonl",
        "data/agent_manual_pages_pdf_rest.jsonl",
    ]
    stats = merge([Path(path) for path in pdf_paths], Path(args.chunk_pages), Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n写出: {args.output}")
    print(f"页面: {stats['records']:,}，文档: {stats['docs']:,}，字符: {stats['chars']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
