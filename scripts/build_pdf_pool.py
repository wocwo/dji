"""大 PDF 多策略解析池。

为什么需要它：
- 40–160MB 的多语言合集会让单个 pypdf 解析卡住；
- 一个文件失败不应该拖垮整批；
- 大文件需要“先尝试精读，超时则用 chunk 页面兜底”。

策略：
1. 每个 PDF 独立子进程，带超时；
2. 最多 N 个并发；
3. 成功 -> 页面级 PDF 记录；
4. 超时/失败 -> 不产出记录，合并阶段自动保留同 doc_id 的 chunk 页面；
5. 后续可在 `_parse_one` 里接 OCR / 表格解析分支。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

from scripts.build_page_corpus import infer_doc_type, infer_model  # noqa: E402

DEFAULT_MODELS = (
    "DJI Mavic 3 Pro,DJI Matrice 30,DJI Matrice 350 RTK,DJI Avata 2,"
    "DJI Inspire 3,DJI Mavic 3 Enterprise,DJI Matrice 4E,DJI Dock 3"
)


def _compact(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _select_pdfs(
    pdf_dir: Path,
    models: set[str],
    include_types: set[str],
    max_size_mb: int,
    *,
    all_models: bool = False,
    exclude_docs: set[str] | None = None,
) -> List[Path]:
    selected: List[Path] = []
    exclude_docs = exclude_docs or set()
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        if infer_doc_type(pdf.name) not in include_types:
            continue
        model_name = infer_model(pdf.name)
        if not all_models and model_name not in models:
            continue
        if all_models and model_name == "DJI Device":
            continue
        if _compact(pdf.stem) in exclude_docs:
            continue
        if max_size_mb and pdf.stat().st_size > max_size_mb * 1024 * 1024:
            continue
        selected.append(pdf)
    return selected


def _parse_one(
    pdf: Path,
    *,
    timeout: int,
    max_pages: int,
    work_dir: Path,
) -> Tuple[Path, List[Dict[str, Any]], str]:
    out = work_dir / (pdf.stem[:80].replace("/", "_") + ".jsonl")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_pdf_pages.py"),
        "--pdf",
        str(pdf),
        "--max-pages",
        str(max_pages),
        "--output",
        str(out),
    ]
    started = time.time()
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return pdf, [], f"timeout>{timeout}s"
    except Exception as exc:
        return pdf, [], f"error:{exc}"
    if not out.exists():
        return pdf, [], "no output"
    records: List[Dict[str, Any]] = []
    try:
        with out.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("text"):
                    records.append(item)
    except Exception as exc:
        return pdf, [], f"read error:{exc}"
    return pdf, records, f"ok {len(records)}p {time.time()-started:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="大 PDF 多策略解析池")
    parser.add_argument("--pdf-dir", default="data/dji_manuals")
    parser.add_argument("--output", default="data/agent_manual_pages_pdf_extra.jsonl")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--include-types",
        default="user_manual,safety,product_information,user_guide",
    )
    parser.add_argument("--max-size-mb", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--all-models", action="store_true", help="不按机型列表限制，处理所有可识别机型")
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=None,
        help="已精读的 PDF 页面 JSONL，按 doc_id 排除",
    )
    args = parser.parse_args()

    models = {item.strip() for item in args.models.split(",") if item.strip()}
    include_types = {item.strip() for item in args.include_types.split(",") if item.strip()}
    exclude_docs: set[str] = set()
    for path in args.exclude_jsonl or []:
        p = Path(path)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("doc_id"):
                    exclude_docs.add(_compact(item["doc_id"]))
    pdfs = _select_pdfs(
        Path(args.pdf_dir),
        models,
        include_types,
        args.max_size_mb,
        all_models=args.all_models,
        exclude_docs=exclude_docs,
    )
    print(f"选中 {len(pdfs)} 本大/缺失机型 PDF，workers={args.workers}，timeout={args.timeout}s", flush=True)

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="dji-pdf-pool-") as tmp:
        work_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    _parse_one,
                    pdf,
                    timeout=args.timeout,
                    max_pages=args.max_pages,
                    work_dir=work_dir,
                ): pdf
                for pdf in pdfs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                pdf, page_records, status = future.result()
                records.extend(page_records)
                if not page_records:
                    failures.append({"file": pdf.name, "reason": status})
                print(
                    f"[{index}/{len(pdfs)}] {pdf.name[:70]} -> {status}；累计 {len(records)} 页",
                    flush=True,
                )

    records.sort(key=lambda item: (item.get("model", ""), item.get("doc_id", ""), int(item.get("page") or 0)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)

    print(
        json.dumps(
            {
                "selected": len(pdfs),
                "records": len(records),
                "docs": len({r.get("doc_id") for r in records}),
                "failures": failures[:20],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"写出: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
