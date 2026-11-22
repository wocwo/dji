"""并行表格抽取池：每个 PDF 独立子进程，超时不影响整批。"""

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


def _pdfs_from_jsonl(paths: List[str]) -> List[Path]:
    seen: set[Path] = set()
    for item in paths:
        source = Path(item)
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = record.get("source") or ""
                pdf = Path(raw)
                if pdf.exists() and pdf.suffix.lower() == ".pdf":
                    seen.add(pdf)
    return sorted(seen)


def _run_one(pdf: Path, work_dir: Path, timeout: int) -> Tuple[Path, List[Dict[str, Any]], str]:
    out = work_dir / (pdf.stem[:80].replace("/", "_") + ".tables.jsonl")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_pdf_tables.py"),
        "--pdf",
        str(pdf),
        "--output",
        str(out),
    ]
    started = time.time()
    try:
        subprocess.run(
            cmd,
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
    records = []
    with out.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("text"):
                records.append(item)
    return pdf, records, f"ok {len(records)} tables {time.time()-started:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="并行表格抽取")
    parser.add_argument("--from-jsonl", action="append", required=True, help="页面语料 JSONL，读取 source 去重")
    parser.add_argument("--output", default="data/agent_manual_tables.jsonl")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    pdfs = _pdfs_from_jsonl(args.from_jsonl)
    print(f"选中 {len(pdfs)} 本 PDF，workers={args.workers}，timeout={args.timeout}s", flush=True)
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="dji-table-pool-") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(_run_one, pdf, work, args.timeout): pdf for pdf in pdfs}
            for index, future in enumerate(as_completed(futures), start=1):
                pdf, table_records, status = future.result()
                records.extend(table_records)
                if not table_records:
                    failures.append({"file": pdf.name, "reason": status})
                print(f"[{index}/{len(pdfs)}] {pdf.name[:65]} -> {status}；累计 {len(records)} 表", flush=True)

    records.sort(key=lambda item: (item.get("model", ""), item.get("doc_id", ""), int(item.get("page") or 0), item.get("table_id", "")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)
    print(json.dumps({"pdfs": len(pdfs), "tables": len(records), "failures": failures[:20]}, ensure_ascii=False, indent=2))
    print(f"写出: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
