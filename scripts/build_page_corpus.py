"""从旧 chunk 语料重建页面级手册库。

背景：
- ``dji_chunks_final.jsonl`` 有 167 万条碎片，很多是标题/表格/页脚/乱码，
  且 ``model`` 只到系列级（DJI Mavic Series），导致引用混机型；
- 直接重解析 1166 份 PDF（9.9GB）需要 pypdf/PyMuPDF，当前环境未安装；
- 这里利用已经抽取的 chunk，按 ``doc_id + page`` 合并为页面级记录，
  清洗乱码/页脚/重复行，并从 PDF 文件名补全机型、版本、语言、文档类型。

输出：``data/agent_manual_pages.jsonl``，供 JsonlKnowledgeBase（BM25）使用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_RE = re.compile(r"[A-Za-z]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPACE_RE = re.compile(r"[ \t]+")
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")
_FOOTER_RE = re.compile(
    r"^\s*\d+\s*©.*(版权所有|All Rights Reserved|Copyright).*$",
    re.IGNORECASE,
)
_TOC_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_VERSION_RE = re.compile(r"(?:^|[_\s-])[vV]\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_LANG_RE = re.compile(r"(zh[-_]?cn|chs|chinese|cn|en|english|multi|lans)", re.IGNORECASE)

_GOOD_PUNCT = set("，。！？；：、（）【】《》“”‘’—…·.,!?;:()[]{}<>%+-/\\|°℃±×÷=~@#&*_'\"")

# 先匹配更具体的型号，避免 "Mini 4" 抢先匹配 "Mini 4 Pro"。
_MODEL_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bMavic\s*4\s*Pro\b", re.I), "DJI Mavic 4 Pro"),
    (re.compile(r"\bMavic\s*3\s*Pro\b", re.I), "DJI Mavic 3 Pro"),
    (re.compile(r"\bMavic\s*3\s*(?:Enterprise|E|T)\b", re.I), "DJI Mavic 3 Enterprise"),
    (re.compile(r"\bMavic\s*3\s*(?:Classic|Cine)\b", re.I), "DJI Mavic 3 Classic"),
    (re.compile(r"\bMavic\s*3\b", re.I), "DJI Mavic 3"),
    (re.compile(r"\bMavic\s*2\s*Pro\b", re.I), "DJI Mavic 2 Pro"),
    (re.compile(r"\bMavic\s*2\s*Zoom\b", re.I), "DJI Mavic 2 Zoom"),
    (re.compile(r"\bMavic\s*2\b", re.I), "DJI Mavic 2"),
    (re.compile(r"\bMini\s*5\s*Pro\b", re.I), "DJI Mini 5 Pro"),
    (re.compile(r"\bMini\s*4\s*Pro\b", re.I), "DJI Mini 4 Pro"),
    (re.compile(r"\bMini\s*4\s*K\b", re.I), "DJI Mini 4K"),
    (re.compile(r"\bMini\s*3\s*Pro\b", re.I), "DJI Mini 3 Pro"),
    (re.compile(r"\bMini\s*3\b", re.I), "DJI Mini 3"),
    (re.compile(r"\bMini\s*2\s*SE\b", re.I), "DJI Mini 2 SE"),
    (re.compile(r"\bMini\s*2\b", re.I), "DJI Mini 2"),
    (re.compile(r"\bAir\s*3\s*S\b", re.I), "DJI Air 3S"),
    (re.compile(r"\bAir\s*3\b", re.I), "DJI Air 3"),
    (re.compile(r"\bAir\s*2\s*S\b", re.I), "DJI Air 2S"),
    (re.compile(r"\bMatrice\s*4\s*D\b", re.I), "DJI Matrice 4D"),
    (re.compile(r"\bMatrice\s*4\s*E\b", re.I), "DJI Matrice 4E"),
    (re.compile(r"\bMatrice\s*4\b", re.I), "DJI Matrice 4"),
    (re.compile(r"\bMatrice\s*30\b", re.I), "DJI Matrice 30"),
    (re.compile(r"\bMatrice\s*300\b", re.I), "DJI Matrice 300 RTK"),
    (re.compile(r"\bMatrice\s*350\b", re.I), "DJI Matrice 350 RTK"),
    (re.compile(r"\bMatrice\s*600\b", re.I), "DJI Matrice 600"),
    (re.compile(r"\bPhantom\s*4\s*Pro\b", re.I), "DJI Phantom 4 Pro"),
    (re.compile(r"\bPhantom\s*4\s*RTK\b", re.I), "DJI Phantom 4 RTK"),
    (re.compile(r"\bPhantom\s*4\b", re.I), "DJI Phantom 4"),
    (re.compile(r"\bPhantom\s*3\b", re.I), "DJI Phantom 3"),
    (re.compile(r"\bInspire\s*3\b", re.I), "DJI Inspire 3"),
    (re.compile(r"\bInspire\s*2\b", re.I), "DJI Inspire 2"),
    (re.compile(r"\bAvata\s*2\b", re.I), "DJI Avata 2"),
    (re.compile(r"\bAvata\b", re.I), "DJI Avata"),
    (re.compile(r"\bNeo\s*2\b", re.I), "DJI Neo 2"),
    (re.compile(r"\bNeo\b", re.I), "DJI Neo"),
    (re.compile(r"\bFlip\b", re.I), "DJI Flip"),
    (re.compile(r"\bFPV\b", re.I), "DJI FPV"),
    (re.compile(r"\bRC\s*Pro\b", re.I), "DJI RC Pro"),
    (re.compile(r"\bRC\s*2\b", re.I), "DJI RC 2"),
    (re.compile(r"\bRC\s*-?\s*N3\b", re.I), "DJI RC-N3"),
    (re.compile(r"\bRC\s*-?\s*N2\b", re.I), "DJI RC-N2"),
    (re.compile(r"\bRC\s*-?\s*N[123]\b", re.I), "DJI RC-N"),
    (re.compile(r"\bGoggles\s*3\b", re.I), "DJI Goggles 3"),
    (re.compile(r"\bGoggles\s*2\b", re.I), "DJI Goggles 2"),
    (re.compile(r"\bGoggles\s*V2\b", re.I), "DJI Goggles V2"),
    (re.compile(r"\bOsmo\s*Action\s*5\s*Pro\b", re.I), "DJI Osmo Action 5 Pro"),
    (re.compile(r"\bOsmo\s*Action\s*4\b", re.I), "DJI Osmo Action 4"),
    (re.compile(r"\bOsmo\s*Pocket\s*3\b", re.I), "DJI Osmo Pocket 3"),
    (re.compile(r"\bRonin\s*4\s*D\b", re.I), "DJI Ronin 4D"),
    (re.compile(r"\bRonin\s*4\b", re.I), "DJI Ronin 4"),
]

_DOC_TYPE_LABELS = {
    "user_manual": "用户手册",
    "user_guide": "用户指南",
    "quick_start": "快速入门指南",
    "safety": "安全指引",
    "product_information": "产品信息",
    "release_notes": "发布说明",
    "other": "文档",
}
_EXCLUDE_TYPES = {
    "case_study", "compatibility", "whitepaper", "brochure", "catalog",
    "poster", "leaflet", "terms", "privacy", "warranty", "datasheet",
}


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", urllib.parse.unquote(value).lower())


def _spaced(value: str) -> str:
    return re.sub(r"[_\-\s]+", " ", urllib.parse.unquote(value)).strip()


def infer_model(name: str) -> str:
    text = _spaced(name)
    for pattern, model in _MODEL_PATTERNS:
        if pattern.search(text):
            return model
    return "DJI Device"


def infer_version(name: str) -> str:
    match = _VERSION_RE.search(_spaced(name))
    return f"v{match.group(1)}" if match else ""


def infer_lang(name: str) -> str:
    text = name.lower()
    if re.search(r"zh[-_]?cn|chs|chinese|中文|简中|_ch\b|_cn\b", text):
        return "zh"
    if re.search(r"english|_en\b|_eng\b", text):
        return "en"
    if re.search(r"multi|lans|30lang|16lang|5lang|7lang", text):
        return "multi"
    return "unknown"


def infer_doc_type(name: str) -> str:
    text = name.lower()
    if "case_study" in text:
        return "case_study"
    if "compatibility" in text:
        return "compatibility"
    if any(key in text for key in ("whitepaper", "brochure", "catalog", "poster", "leaflet")):
        return "brochure"
    if any(key in text for key in ("terms", "privacy", "warranty", "datasheet")):
        return "terms"
    if "release_notes" in text or "release note" in text:
        return "release_notes"
    if "user_manual" in text or "用户手册" in text or "user manual" in text:
        return "user_manual"
    if "quick_start" in text or "quick start" in text:
        return "quick_start"
    if "safety" in text or "安全" in text:
        return "safety"
    if "product_information" in text or "product_info" in text or "产品信息" in text:
        return "product_information"
    if "user_guide" in text or "user guide" in text or "使用说明" in text or "说明书" in text:
        return "user_guide"
    return "other"


def build_title(name: str, model: str, version: str, doc_type: str) -> str:
    label = _DOC_TYPE_LABELS.get(doc_type, "文档")
    if model != "DJI Device":
        return f"{model} {label}{(' ' + version) if version else ''}"
    cleaned = _spaced(re.sub(r"[vV]\d+(?:\.\d+)*", "", name))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    return cleaned[:100] or name


def clean_text(raw: str) -> str:
    """清掉 NUL、页脚、目录行和明显的乱码行。"""
    text = _CONTROL_RE.sub(" ", raw or "")
    text = text.replace("\x00", " ").replace("\ufffd", " ")
    lines: List[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = _SPACE_RE.sub(" ", line).strip()
        if not line or _PAGE_NUM_RE.match(line) or _FOOTER_RE.match(line) or _TOC_RE.search(line):
            continue
        if "版权所有" in line or "All Rights Reserved" in line:
            continue
        if len(line) < 2 and not _CJK_RE.search(line):
            continue
        good = sum(
            1
            for ch in line
            if _CJK_RE.match(ch) or ch.isalnum() or ch in _GOOD_PUNCT
        )
        if good / max(1, len(line)) < 0.62:
            continue
        key = re.sub(r"\s+", "", line)
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def usable_text(text: str, *, min_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    cjk = len(_CJK_RE.findall(text))
    ascii_letters = len(_ASCII_RE.findall(text))
    return cjk >= 15 or ascii_letters >= 80


def usable_chunk(text: str) -> bool:
    """chunk 级只排垃圾；短句要保留，否则整页会被误删。"""
    if len(text) < 8:
        return False
    cjk = len(_CJK_RE.findall(text))
    ascii_letters = len(_ASCII_RE.findall(text))
    return cjk >= 3 or ascii_letters >= 10


def split_long(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[。！？!?；;])", text)
    chunks: List[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if len(buffer) + len(part) > max_chars and buffer:
            chunks.append(buffer)
            buffer = ""
        buffer += part
    if buffer:
        chunks.append(buffer)
    if not chunks:
        chunks = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return chunks


def _build_pdf_index(pdf_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not pdf_dir.exists():
        return index
    for pdf in pdf_dir.glob("*.pdf"):
        index[_compact(pdf.stem)] = pdf
    return index


def _match_pdf(doc_id: str, index: Dict[str, Path]) -> Optional[Path]:
    key = _compact(doc_id)
    if key in index:
        return index[key]
    # 旧 doc_id 可能被截断/多了后缀；选公共前缀最长的候选，避免
    # "某机型 Quick Start" 抢先匹配 "同一机型 User Manual"。
    best: Optional[Path] = None
    best_len = 0
    for candidate, pdf in index.items():
        if key and (candidate.startswith(key) or key.startswith(candidate)):
            common = 0
            for left, right in zip(key, candidate):
                if left != right:
                    break
                common += 1
            if common > best_len:
                best = pdf
                best_len = common
    return best


def _section_of(text: str) -> str:
    first = text.splitlines()[0].strip() if text else ""
    if 4 <= len(first) <= 60 and not first.endswith(("。", "！", "？", ".", "!", "?")):
        return first
    return ""


def build(
    *,
    source: Path,
    pdf_dir: Path,
    output: Path,
    max_chars: int,
    min_chars: int,
    include_types: Optional[set[str]] = None,
) -> Dict[str, Any]:
    pdf_index = _build_pdf_index(pdf_dir)
    pages: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    doc_names: Dict[str, str] = {}
    total = 0
    kept_chunks = 0

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            total += 1
            try:
                record = json.loads(line)
            except Exception:
                continue
            doc_id = urllib.parse.unquote(str(record.get("doc_id") or "")).strip()
            if not doc_id:
                continue
            page = record.get("page")
            try:
                page = int(page)
            except (TypeError, ValueError):
                continue
            text = clean_text(str(record.get("text") or ""))
            if not usable_chunk(text):
                continue
            pages[(doc_id, page)].append(text)
            doc_names.setdefault(doc_id, doc_id)
            kept_chunks += 1

    records: List[Dict[str, Any]] = []
    used_docs: set[str] = set()
    seen_pages: set[str] = set()
    for (doc_id, page), chunks in pages.items():
        pdf = _match_pdf(doc_id, pdf_index)
        name = pdf.stem if pdf else doc_id
        doc_type = infer_doc_type(name)
        if doc_type in _EXCLUDE_TYPES:
            continue
        if include_types and doc_type not in include_types:
            continue
        model = infer_model(name)
        version = infer_version(name)
        lang = infer_lang(name)
        title = build_title(name, model, version, doc_type)

        merged = "\n".join(chunks)
        # 去掉跨 chunk 的重复行
        lines: List[str] = []
        seen_lines: set[str] = set()
        for line in merged.splitlines():
            key = re.sub(r"\s+", "", line)
            if not key or key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(line)
        merged = "\n".join(lines).strip()
        if not usable_text(merged, min_chars=min_chars):
            continue

        for part_index, piece in enumerate(split_long(merged, max_chars)):
            piece = piece.strip()
            if not usable_text(piece, min_chars=min_chars):
                continue
            digest = hashlib.sha1(
                f"{doc_id}|{page}|{part_index}|{piece[:200]}".encode("utf-8")
            ).hexdigest()[:10]
            chunk_id = f"{_compact(doc_id)[:80]}-p{page}-{part_index}-{digest}"
            if chunk_id in seen_pages:
                continue
            seen_pages.add(chunk_id)
            try:
                source_path = str(pdf.resolve().relative_to(REPO_ROOT))
            except ValueError:
                source_path = str(pdf)
            records.append(
                {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "doc_id": doc_id,
                    "title": title,
                    "section": _section_of(piece),
                    "page": page,
                    "model": model,
                    "version": version,
                    "lang": lang,
                    "doc_type": doc_type,
                    "type": "page",
                    "source_kind": "chunk",
                    "source": source_path,
                    "keywords": [],
                }
            )
            used_docs.add(doc_id)

    records.sort(key=lambda item: (item["doc_id"], int(item["page"]), item["chunk_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    model_counts = Counter(record["model"] for record in records)
    type_counts = Counter(record["doc_type"] for record in records)
    lang_counts = Counter(record["lang"] for record in records)
    return {
        "source_chunks": total,
        "kept_chunks": kept_chunks,
        "pages": len(pages),
        "records": len(records),
        "docs": len(used_docs),
        "chars": sum(len(record["text"]) for record in records),
        "models": model_counts.most_common(20),
        "types": type_counts.most_common(),
        "langs": lang_counts.most_common(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从旧 chunk 重建页面级手册库")
    parser.add_argument("--source", default="dji_chunks_final.jsonl")
    parser.add_argument("--pdf-dir", default="data/dji_manuals")
    parser.add_argument("--output", default="data/agent_manual_pages.jsonl")
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--min-chars", type=int, default=80)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"源文件不存在: {source}", file=sys.stderr)
        return 1

    stats = build(
        source=source,
        pdf_dir=Path(args.pdf_dir),
        output=Path(args.output),
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n写出: {args.output}")
    print(f"记录: {stats['records']:,}，文档: {stats['docs']:,}，字符: {stats['chars']:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
