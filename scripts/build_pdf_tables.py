"""从 PDF 抽取表格，生成独立的 table 记录。

输出记录同时保留：
- Markdown 文本（给 BM25 / LLM）
- columns / rows（给程序校验）
- 页码、机型、版本、表格序号
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_VENDOR = REPO_ROOT / ".vendor"
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from scripts.build_page_corpus import (  # noqa: E402
    build_title,
    clean_text,
    infer_doc_type,
    infer_lang,
    infer_model,
    infer_version,
)

import pdfplumber  # noqa: E402


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def _table_to_markdown(header: List[str], rows: List[List[str]]) -> str:
    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(cell) for cell in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        cells = [esc(cell) for cell in row[: len(header)]]
        cells += [""] * (len(header) - len(cells))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_tables(
    pdf_path: Path,
    *,
    max_pages: int = 400,
    max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    doc_id = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_") or pdf_path.stem
    model = infer_model(pdf_path.name)
    version = infer_version(pdf_path.name)
    lang = infer_lang(pdf_path.name)
    doc_type = infer_doc_type(pdf_path.name)
    title = build_title(pdf_path.name, model, version, doc_type)

    records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page_index, page in enumerate(pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                continue
            for table_index, table in enumerate(tables, start=1):
                if not table:
                    continue
                cleaned = [[_clean_cell(cell) for cell in row] for row in table]
                # 去掉全空行 / 全空列
                cleaned = [row for row in cleaned if any(cell for cell in row)]
                if len(cleaned) < 2:
                    continue
                width = max(len(row) for row in cleaned)
                if width < 2:
                    continue
                cleaned = [row + [""] * (width - len(row)) for row in cleaned]
                non_empty = sum(1 for row in cleaned for cell in row if cell)
                if non_empty < 4:
                    continue

                header = cleaned[0]
                if not any(header):
                    header = [f"列{i + 1}" for i in range(width)]
                body = cleaned[1:]
                if not body:
                    continue
                markdown = _table_to_markdown(header, body)
                if len(markdown) > max_chars:
                    markdown = markdown[:max_chars] + "\n…（表格已截断）"
                fingerprint = hashlib.sha1(
                    f"{doc_id}|{page_index}|{markdown[:300]}".encode("utf-8")
                ).hexdigest()[:10]
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                table_id = f"{doc_id[:70]}-p{page_index}-t{table_index}"
                records.append(
                    {
                        "chunk_id": table_id,
                        "table_id": table_id,
                        "text": f"{model} {title} 第{page_index}页 表{table_index}\n{markdown}",
                        "doc_id": doc_id,
                        "title": title,
                        "section": f"表{table_index}",
                        "page": page_index,
                        "model": model,
                        "version": version,
                        "lang": lang,
                        "doc_type": doc_type,
                        "type": "table",
                        "source_kind": "pdf_table",
                        "source": str(pdf_path.resolve().relative_to(REPO_ROOT)),
                        "columns": header,
                        "rows": body,
                        "n_rows": len(body),
                        "n_cols": width,
                        "keywords": [],
                    }
                )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="从 PDF 抽取表格")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pages", type=int, default=400)
    parser.add_argument("--max-chars", type=int, default=8000)
    args = parser.parse_args()
    records = extract_tables(
        Path(args.pdf),
        max_pages=args.max_pages,
        max_chars=args.max_chars,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{Path(args.pdf).name}: {len(records)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
