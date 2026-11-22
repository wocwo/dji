"""从核心机型 PDF 直接构建页面级手册库。

只处理文件名能识别出具体机型的文档，排除案例/兼容性/条款等非手册内容；
每个 PDF 页面生成一条记录，保留页码、机型、版本、文档类型和源文件。

用法::

    python scripts/build_pdf_pages.py
    python scripts/build_pdf_pages.py --max-chars 4000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    _EXCLUDE_TYPES,
    _compact,
    build_title,
    clean_text,
    infer_doc_type,
    infer_lang,
    infer_model,
    infer_version,
    split_long,
    usable_text,
)

from pypdf import PdfReader  # noqa: E402

_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[^\s].{0,48}$")


def _section_of_lines(lines: List[str], previous: str) -> str:
    for line in lines[:8]:
        text = line.strip()
        if _HEADING_RE.match(text):
            return text[:60]
        if 4 <= len(text) <= 40 and text.endswith(("概述", "电池", "升级", "校准", "飞行", "安全", "设置", "返航")):
            return text
    return previous


def _extract_pdf(
    path: Path,
    *,
    max_chars: int,
    min_chars: int,
    max_pages: int = 0,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    doc_id = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_") or path.stem
    model = infer_model(path.name)
    version = infer_version(path.name)
    lang = infer_lang(path.name)
    doc_type = infer_doc_type(path.name)
    title = build_title(path.name, model, version, doc_type)

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except Exception as exc:
        return [], f"打开失败: {exc}"

    records: List[Dict[str, Any]] = []
    section = ""
    page_total = len(reader.pages)
    if max_pages and page_total > max_pages:
        page_total = max_pages
    for page_index in range(page_total):
        try:
            raw = reader.pages[page_index].extract_text() or ""
        except Exception as exc:  # pragma: no cover - 个别损坏页
            print(f"[warn] {path.name} 第 {page_index + 1} 页失败: {exc}")
            continue
        text = clean_text(raw)
        if not usable_text(text, min_chars=min_chars):
            continue
        lines = text.splitlines()
        section = _section_of_lines(lines, section)
        for part_index, piece in enumerate(split_long(text, max_chars)):
            piece = piece.strip()
            if not usable_text(piece, min_chars=min_chars):
                continue
            chunk_id = f"{doc_id[:80]}-p{page_index + 1}-{part_index:02d}"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "doc_id": doc_id,
                    "title": title,
                    "section": section,
                    "page": page_index + 1,
                    "model": model,
                    "version": version,
                    "lang": lang,
                    "doc_type": doc_type,
                    "type": "page",
                    "source_kind": "pdf",
                    "source": str(path.resolve().relative_to(REPO_ROOT)),
                    "keywords": [],
                }
            )
    return records, None


def build(
    *,
    pdf_dir: Path,
    output: Path,
    include_types: Set[str],
    max_chars: int,
    min_chars: int,
    max_size_mb: int = 0,
    max_pages: int = 0,
    models: Optional[Set[str]] = None,
    explicit_pdfs: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    pdfs = explicit_pdfs if explicit_pdfs is not None else sorted(pdf_dir.glob("*.pdf"))
    selected: List[Path] = []
    skipped_large: List[str] = []
    for pdf in pdfs:
        doc_type = infer_doc_type(pdf.name)
        if explicit_pdfs is None:
            if doc_type in _EXCLUDE_TYPES:
                continue
            if doc_type not in include_types:
                continue
            model_name = infer_model(pdf.name)
            if model_name == "DJI Device":
                continue
            if models and model_name not in models:
                continue
        if max_size_mb and pdf.stat().st_size > max_size_mb * 1024 * 1024:
            skipped_large.append(pdf.name)
            continue
        selected.append(pdf)

    print(
        f"PDF 总数 {len(pdfs)}，选中 {len(selected)} 本，"
        f"约 {sum(p.stat().st_size for p in selected)/1024/1024:.0f} MB，"
        f"跳过超大 {len(skipped_large)} 本",
        flush=True,
    )
    for name in skipped_large[:10]:
        print(f"  [skip large] {name}", flush=True)

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    started = time.time()
    for index, pdf in enumerate(selected, start=1):
        page_records, error = _extract_pdf(
            pdf,
            max_chars=max_chars,
            min_chars=min_chars,
            max_pages=max_pages,
        )
        if error:
            errors.append({"file": pdf.name, "error": error})
        records.extend(page_records)
        if index <= 5 or index % 20 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {pdf.name[:70]} "
                f"+{len(page_records)} 页，累计 {len(records)} 页，"
                f"{time.time() - started:.0f}s",
                flush=True,
            )

    records.sort(key=lambda item: (item["model"], item["doc_id"], int(item["page"]), item["chunk_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)

    from collections import Counter

    models = Counter(record["model"] for record in records)
    types = Counter(record["doc_type"] for record in records)
    langs = Counter(record["lang"] for record in records)
    docs = len({record["doc_id"] for record in records})
    return {
        "selected_pdfs": len(selected),
        "records": len(records),
        "docs": docs,
        "chars": sum(len(record["text"]) for record in records),
        "models": models.most_common(30),
        "types": types.most_common(),
        "langs": langs.most_common(),
        "errors": errors[:20],
        "elapsed_s": round(time.time() - started, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从 PDF 构建页面级手册库")
    parser.add_argument("--pdf-dir", default="data/dji_manuals")
    parser.add_argument("--output", default="data/agent_manual_pages.jsonl")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument(
        "--include-types",
        default="user_manual,user_guide,safety,quick_start,product_information,release_notes,other",
    )
    parser.add_argument("--max-size-mb", type=int, default=0, help="跳过超过该大小的 PDF；0 不限制")
    parser.add_argument("--max-pages", type=int, default=0, help="每本最多解析页数；0 不限制")
    parser.add_argument("--models", default="", help="只解析指定机型，逗号分隔；空表示全部")
    parser.add_argument("--pdf", action="append", default=None, help="显式 PDF 路径（可重复），跳过类型/机型筛选")
    args = parser.parse_args()

    include_types = {item.strip() for item in args.include_types.split(",") if item.strip()}
    stats = build(
        pdf_dir=Path(args.pdf_dir),
        output=Path(args.output),
        include_types=include_types,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
        max_size_mb=args.max_size_mb,
        max_pages=args.max_pages,
        models={item.strip() for item in args.models.split(",") if item.strip()} or None,
        explicit_pdfs=[Path(item) for item in args.pdf] if args.pdf else None,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n写出: {args.output}")
    print(f"页面: {stats['records']:,}，文档: {stats['docs']:,}，字符: {stats['chars']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
