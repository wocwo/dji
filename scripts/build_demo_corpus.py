"""构建 Agent 离线演示语料。

优先从 ``data/dji_manuals`` 的官方 PDF 用户手册中抽取高质量文本；
没有 PDF 解析依赖时，可退化为从 Chroma 集合合并片段。

生成：``data/agent_demo_corpus.jsonl``，供 ``python -m agent demo``、
``python -m agent eval`` 以及测试使用。

用法::

    python scripts/build_demo_corpus.py                 # 从三本用户手册抽取
    python scripts/build_demo_corpus.py --source chroma # 从 chroma_final 合并
    python scripts/build_demo_corpus.py --pdf <a.pdf> --pdf <b.pdf>

PDF 依赖（仅生成语料时需要）::

    pip install pypdf cryptography --target .vendor
    python scripts/build_demo_corpus.py

脚本会自动把仓库根目录的 ``.vendor`` 加入 ``sys.path``（若存在）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


DEFAULT_PDFS = [
    "data/dji_manuals/DJI_Mavic_4_Pro_User_Manual_v1.2_zh-cn.pdf",
    "data/dji_manuals/DJI_Air_3S_User_Manual_v1.0_zh-cn.pdf",
    "data/dji_manuals/DJI_Mini_4_Pro_User_Manual_chs.pdf",
]

DEFAULT_TERMS = [
    "智能飞行电池", "低温", "电池保养", "充电", "固件升级", "固件回退",
    "遥控器对频", "起飞", "返航", "避障", "图传", "存储卡", "桨叶",
    "电机", "指南针校准", "IMU", "飞行限制", "智能跟随", "拍照", "录像",
    "Mavic 4 Pro", "Mini 4 Pro", "Air 3S", "DJI Assistant",
]

TOPIC_KEYWORDS = {
    "电池": ["电池", "电量", "充电"],
    "固件": ["固件", "升级"],
    "对频": ["对频", "遥控器"],
    "飞行": ["起飞", "降落", "返航", "飞行限制"],
    "安全": ["安全", "警告", "注意"],
    "影像": ["拍照", "录像", "相机", "云台"],
    "图传": ["图传", "信号", "频段"],
    "校准": ["校准", "指南针", "IMU"],
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_NUL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_FOOTER_RE = re.compile(r"^\s*\d+\s*©.*(版权所有|All Rights Reserved).*$", re.IGNORECASE)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,3}\s*$")
_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[^\s].{0,48}$")
_TOC_LINE_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_BULLET_PREFIXES = ("•", "◆", "●", "▪", "·", "-", "*")


# ---------------------------------------------------------------------------
# PDF 抽取
# ---------------------------------------------------------------------------

def _normalize_title(path: Path) -> Tuple[str, str, str]:
    stem = path.stem
    model = "DJI"
    for candidate in ("Mavic 4 Pro", "Air 3S", "Mini 4 Pro", "Mavic 3", "Air 3", "Mini 4"):
        if candidate.lower().replace(" ", "_") in stem.lower() or candidate.lower().replace(" ", "") in stem.lower():
            model = f"DJI {candidate}"
            break
    version = ""
    match = re.search(r"[vV](\d+(?:\.\d+)*)", stem)
    if match:
        version = f" v{match.group(1)}"
    title = f"{model} 用户手册{version}" if model != "DJI" else stem
    doc_id = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return doc_id, title, model


def _is_cjk(char: str) -> bool:
    return bool(_CJK_RE.match(char))


def _join_line(current: str, line: str) -> str:
    if not current:
        return line
    if _is_cjk(current[-1]) and _is_cjk(line[0]):
        return current + line
    if current[-1] in "，。！？；：、…—）】》" or line[0] in "，。！？；：、…—）》】":
        return current + line
    return current + " " + line


def _clean_page(text: str) -> List[Tuple[str, str]]:
    """返回 [(kind, text)]，kind 为 H（标题）/B（项目符号）/T（正文）。"""
    blocks: List[Tuple[str, str]] = []
    current = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(("T", current))
                current = ""
            continue
        if _FOOTER_RE.match(line) or _PAGE_NUM_RE.match(line):
            continue
        if _TOC_LINE_RE.search(line):
            continue
        if _HEADING_RE.match(line):
            if current:
                blocks.append(("T", current))
                current = ""
            blocks.append(("H", line))
            continue
        if line.startswith(_BULLET_PREFIXES):
            if current:
                blocks.append(("T", current))
                current = ""
            blocks.append(("B", line))
            continue
        current = _join_line(current, line)
        if len(current) >= 500:
            blocks.append(("T", current))
            current = ""
    if current:
        blocks.append(("T", current))
    return blocks


def _split_long(text: str, max_chars: int) -> List[str]:
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
        for i in range(0, len(text), max_chars):
            chunks.append(text[i:i + max_chars])
    return chunks


def _topics_for(text: str) -> List[str]:
    topics: List[str] = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
    return topics


def extract_pdf(
    path: Path,
    *,
    per_doc: int,
    max_chars: int,
    min_chars: int,
) -> List[Dict[str, Any]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    doc_id, title, model = _normalize_title(path)
    records: List[Dict[str, Any]] = []
    buffer = ""
    buffer_page = 1
    section = ""

    def flush() -> None:
        nonlocal buffer
        text = buffer.strip()
        buffer = ""
        if any(
            marker in text
            for marker in ("版权和所有权", "快速搜索关键词", "阅读提示符号说明", "打印文档")
        ):
            return
        if len(text) < min_chars:
            return
        if len(_CJK_RE.findall(text)) < 20:
            return
        for piece in _split_long(text, max_chars):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            records.append(
                {
                    "chunk_id": f"{doc_id}-p{buffer_page}-{len(records):04d}",
                    "text": piece,
                    "doc_id": doc_id,
                    "title": title,
                    "section": section,
                    "page": buffer_page,
                    "model": model,
                    "type": "文本",
                    "lang": "zh",
                    "keywords": [],
                    "topics": _topics_for(piece),
                }
            )

    page_count = len(reader.pages)
    for page_index in range(page_count):
        if page_index < 2:
            # 封面、版权页
            continue
        try:
            text = reader.pages[page_index].extract_text() or ""
        except Exception as exc:  # pragma: no cover - 个别损坏页
            print(f"[warn] {path.name} 第 {page_index + 1} 页解析失败: {exc}")
            continue
        text = _NUL_RE.sub("", text)
        if len(text.strip()) < 20:
            continue
        # 目录页：大量 “...... 12” 行（在清洗前判断）
        toc_lines = sum(1 for line in text.splitlines() if _TOC_LINE_RE.search(line.strip()))
        if toc_lines > 5:
            continue
        blocks = _clean_page(text)

        for kind, line in blocks:
            if kind == "H":
                if buffer:
                    flush()
                section = line[:60]
                continue
            if kind == "B":
                if buffer and len(buffer) + len(line) + 1 > max_chars:
                    flush()
                buffer = (buffer + "\n" + line) if buffer else line
                buffer_page = page_index + 1
                continue
            if buffer and len(buffer) + len(line) > max_chars:
                flush()
            if not buffer:
                buffer_page = page_index + 1
            buffer = _join_line(buffer, line)
            if len(buffer) >= max_chars:
                flush()
        if len(records) >= per_doc:
            break

    if buffer:
        flush()
    return records[:per_doc]


def build_from_pdfs(
    pdf_paths: Sequence[str],
    *,
    per_doc: int,
    max_docs: int,
    max_chars: int,
    min_chars: int,
) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for pdf in pdf_paths:
        path = Path(pdf)
        if not path.exists():
            print(f"[warn] 文件不存在，跳过: {path}")
            continue
        print(f"[pdf] {path.name} ...")
        try:
            records = extract_pdf(
                path,
                per_doc=per_doc,
                max_chars=max_chars,
                min_chars=min_chars,
            )
        except ImportError as exc:
            print(
                "缺少 PDF 解析依赖。请执行：\n"
                "  pip install pypdf cryptography --target .vendor\n"
                f"原始错误: {exc}",
                file=sys.stderr,
            )
            return []
        except Exception as exc:
            print(f"[warn] {path.name} 解析失败: {exc}")
            continue
        kept = 0
        for record in records:
            fingerprint = re.sub(r"\s+", "", record["text"])[:180]
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            all_records.append(record)
            kept += 1
        print(f"      +{kept} 片段（累计 {len(all_records)}）")
        if len(all_records) >= max_docs:
            break
    return all_records[:max_docs]


# ---------------------------------------------------------------------------
# Chroma 回退：按文档 + 页码合并碎片
# ---------------------------------------------------------------------------

def _clean_chroma_text(raw: str) -> str:
    text = _NUL_RE.sub(" ", raw or "")
    lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[|\s:\-\d]+", line):
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            cells = [c for c in cells if c and not re.fullmatch(r"[\s:\-\d]+", c)]
            if not cells:
                continue
            line = " ".join(cells)
        line = re.sub(r"[ \t]+", " ", line).strip()
        if len(re.findall(r"[\u4e00-\u9fffA-Za-z]", line)) >= 2:
            lines.append(line)
    return "\n".join(lines)


def build_from_chroma(
    chroma_path: str,
    collection_name: str,
    *,
    per_term: int,
    max_docs: int,
    min_chars: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    import chromadb

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    print(f"[open] {chroma_path} / {collection_name}：{collection.count()} 条向量")

    records: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    for index, term in enumerate(DEFAULT_TERMS, start=1):
        try:
            result = collection.get(
                where_document={"$contains": term},
                limit=max(per_term * 10, 50),
                include=["documents", "metadatas"],
            )
        except Exception as exc:  # pragma: no cover
            print(f"[warn] 检索词 {term!r} 失败: {exc}")
            continue
        groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
        for chunk_id, document, metadata in zip(
            result.get("ids") or [],
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            metadata = metadata or {}
            text = _clean_chroma_text(document)
            if not text:
                continue
            key = (str(metadata.get("doc_id") or ""), metadata.get("page"))
            groups.setdefault(key, []).append({"id": str(chunk_id), "text": text, "metadata": metadata})

        kept = 0
        for (doc_id, page), items in groups.items():
            merged = "\n".join(item["text"] for item in items)
            for piece in _split_long(merged, max_chars):
                piece = piece.strip()
                if len(piece) < min_chars or len(_CJK_RE.findall(piece)) < 12:
                    continue
                fingerprint = re.sub(r"\s+", "", piece)[:180]
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                metadata = items[0]["metadata"]
                title = str(metadata.get("title") or "").strip()
                if title in ("无标题", "None", "null"):
                    title = doc_id
                chunk_id = f"{doc_id}-p{page}-{len(records):04d}"
                records[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": piece,
                    "doc_id": doc_id,
                    "title": title,
                    "section": str(metadata.get("section") or "")[:120],
                    "page": page,
                    "model": str(metadata.get("model") or ""),
                    "type": str(metadata.get("type") or "文本"),
                    "lang": str(metadata.get("lang") or ""),
                    "keywords": [],
                    "topics": _topics_for(piece),
                }
                kept += 1
                if kept >= per_term:
                    break
            if kept >= per_term:
                break
        print(f"[{index}/{len(DEFAULT_TERMS)}] {term}: +{kept}（累计 {len(records)}）")
        if len(records) >= max_docs:
            break
    return list(records.values())[:max_docs]


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Agent 离线演示语料")
    parser.add_argument("--source", choices=["auto", "pdf", "chroma"], default="auto")
    parser.add_argument("--pdf", action="append", default=None, help="PDF 路径（可重复）")
    parser.add_argument("--chroma", default="./chroma_final", help="Chroma 目录")
    parser.add_argument("--collection", default="uav_manual_db")
    parser.add_argument("--output", default="data/agent_demo_corpus.jsonl")
    parser.add_argument("--per-doc", type=int, default=120, help="每本手册最多片段数")
    parser.add_argument("--max-docs", type=int, default=360, help="总片段上限")
    parser.add_argument("--min-chars", type=int, default=120, help="片段最少字符数")
    parser.add_argument("--max-chars", type=int, default=800, help="片段最大字符数")
    args = parser.parse_args()

    pdf_paths = args.pdf or DEFAULT_PDFS
    use_pdf = args.source == "pdf" or (
        args.source == "auto" and any(Path(p).exists() for p in pdf_paths)
    )

    if use_pdf:
        records = build_from_pdfs(
            pdf_paths,
            per_doc=args.per_doc,
            max_docs=args.max_docs,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
        )
    else:
        records = build_from_chroma(
            args.chroma,
            args.collection,
            per_term=max(5, args.per_doc // 6),
            max_docs=args.max_docs,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

    if not records:
        print("未生成任何片段。请检查 PDF 路径或 Chroma 集合。", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    docs = len({record["doc_id"] for record in records})
    size_kb = output.stat().st_size / 1024
    avg_len = sum(len(record["text"]) for record in records) / len(records)
    topics = sorted({topic for record in records for topic in record["topics"]})
    print("\n" + "=" * 60)
    print(f"已写出：{output}（{size_kb:.1f} KB）")
    print(f"片段数：{len(records)}，来源文档：{docs}，平均长度：{avg_len:.0f} 字符")
    print(f"主题：{', '.join(topics) if topics else '(无)'}")
    print("=" * 60)
    print("\n示例片段：")
    print(records[0]["text"][:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
