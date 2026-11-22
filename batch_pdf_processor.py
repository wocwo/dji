"""
DJI 手册 PDF 批量处理：PyMuPDF + Camelot 抽取文本/表格，按 token 窗口切分并写出 JSONL。

依赖 fitz(PyMuPDF)、camelot、tiktoken。输出路径见下方 INPUT_FOLDER / OUTPUT_JSONL。
"""

import fitz
import camelot
import tiktoken
import json
import uuid
import re
import os
from pathlib import Path
from collections import defaultdict

# ====================== 【全局配置】 ======================
INPUT_FOLDER = "data/dji_manuals"
OUTPUT_JSONL = "dji_chunks_final.jsonl"
REPORT_PATH = "process_report_final.txt"

TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
MIN_TOKENS = 200
MAX_TOKENS = 512
MAX_LEN_HARD = 700

# 术语词典（全系列通用）
DJI_TERMS = {
    "云台", "IMU", "GPS", "RTK", "图传", "飞控", "智能返航", "APAS", "ADS-B",
    "校准", "固件升级", "遥控器", "电池", "避障", "发射器", "接收器", "TX", "RX",
    "蓝牙", "射频", "充电盒", "续航", "分辨率", "Micro SD", "对频", "开机", "关机",
    "变焦", "对焦", "快门", "ISO", "白平衡", "D-LOG", "D-Cinelike", "FOV",
    "SAR", "FCC", "CE", "ROHS", "GFSK", "EIRP"
}

# 警告标识
WARNING_WORDS = ["警告", "注意", "警示", "禁止", "请勿", "必须", "危险"]

# 标题正则（自动识别章节）
REGEX_H1 = re.compile(r'^(第.{1,6}章|前言|安全|产品概述|安装|操作|规格|参数|故障|免责|附录)', re.I)
REGEX_H2 = re.compile(r'^\d+\.\d+[^.]|^[一二三四五六七八九十]{1,3}\s*.|^概述|简介|物品清单|部件名称', re.I)
REGEX_H3 = re.compile(r'^\d+\.\d+\.\d+|^\(\d\)|^[①-⑩]', re.I)

# 语言检测正则
LANG_PATTERNS = {
    "zh": re.compile(r'[\u4e00-\u9fff]'),
    "en": re.compile(r'[a-zA-Z]'),
    "ko": re.compile(r'[\uac00-\ud7af]'),
    "ja": re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')
}

# ====================== 工具函数 ======================
def list_all_pdfs(folder):
    paths = []
    for r, ds, fs in os.walk(folder):
        for f in fs:
            if f.lower().endswith(".pdf"):
                paths.append(str(Path(r) / f))
    return paths

def clean_text(text):
    text = re.sub(r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页", "", text)
    text = re.sub(r"©\s*\d{4}\s*大疆创新", "", text)
    text = re.sub(r"图\s*\d+[-.]\d+.*", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def count_tokens(text):
    return len(TOKEN_ENCODER.encode(text))

def get_doc_id(filename):
    return Path(filename).stem.lower().replace(" ", "_")

def detect_model(name):
    name = name.lower()
    if "mini" in name: return "DJI Mini Series"
    if "mavic" in name: return "DJI Mavic Series"
    if "air" in name: return "DJI Air Series"
    if "mic" in name: return "DJI Mic Series"
    if "zenmuse" in name: return "DJI Zenmuse Series"
    if "rc" in name or "remote" in name: return "DJI RC Series"
    if "matrice" in name: return "DJI Matrice Series"
    if "inspire" in name: return "DJI Inspire Series"
    return "DJI Device"

def detect_lang(text):
    scores = {}
    for lang, pat in LANG_PATTERNS.items():
        scores[lang] = len(pat.findall(text))
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "unknown"

def classify_type(text):
    if any(w in text for w in WARNING_WORDS):
        return "警告"
    if re.search(r"参数|规格|尺寸|重量|容量|频率|功率|dBm|SAR|ROHS|FCC|CE", text):
        return "参数"
    if re.search(r"步骤|1\.|2\.|首先|然后|点击|连接|长按|插入|安装|校准|升级|对频", text):
        return "步骤"
    if re.search(r"故障|无法|异常|失灵|报错|失败|丢失|解决|原因", text):
        return "故障"
    if re.search(r"是指|用于|定义|支持|功能|概述|简介", text):
        return "定义"
    if re.search(r"对比|vs|相比|差异|优于|区别", text):
        return "对比"
    return "文本"

def extract_keywords(text):
    ks = [t for t in DJI_TERMS if t in text]
    return list(set(ks))[:5]

# ====================== 标题层级解析（核心增强） ======================
def classify_title_level(text):
    if REGEX_H1.search(text):
        return 1
    if REGEX_H2.search(text):
        return 2
    if REGEX_H3.search(text):
        return 3
    return 0

# ====================== 单PDF处理（完整版） ======================
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    doc_id = get_doc_id(pdf_path)
    model = detect_model(pdf_path)
    title_stack = []

    for page_num, page in enumerate(doc, 1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    clean = clean_text(text)
                    if len(clean) < 2:
                        continue

                    # 标题层级
                    lvl = classify_title_level(clean)
                    if lvl == 1:
                        title_stack = [clean]
                    elif lvl == 2:
                        title_stack = title_stack[:1] + [clean]
                    elif lvl == 3:
                        title_stack = title_stack[:2] + [clean]

                    # 章节信息
                    section = " > ".join(title_stack)
                    title = title_stack[-1] if title_stack else "无标题"
                    lang = detect_lang(clean)

                    # 构建chunk
                    chunk = {
                        "doc_id": doc_id,
                        "model": model,
                        "title": title,
                        "section": section,
                        "text": clean,
                        "type": classify_type(clean),
                        "keywords": extract_keywords(clean),
                        "token_count": count_tokens(clean),
                        "lang": lang,
                        "page": page_num,
                        "chunk_id": str(uuid.uuid4())[:8]
                    }
                    chunks.append(chunk)

    # 表格提取
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for tbl in tables:
            md = tbl.df.to_markdown(index=False)
            chunk = {
                "doc_id": doc_id,
                "model": model,
                "title": "参数表",
                "section": "规格参数",
                "text": md,
                "type": "参数",
                "keywords": extract_keywords(md),
                "token_count": count_tokens(md),
                "lang": detect_lang(md),
                "page": 0,
                "chunk_id": str(uuid.uuid4())[:8]
            }
            chunks.append(chunk)
    except Exception:
        pass

    return chunks

# ====================== 批量处理 ======================
def batch_process():
    pdfs = list_all_pdfs(INPUT_FOLDER)
    print(f"📂 发现 PDF 数量：{len(pdfs)}")
    all_chunks = []
    for i, f in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] 处理：{Path(f).name}")
        chunks = process_pdf(f)
        all_chunks.extend(chunks)
    return all_chunks

# ====================== 质量报告 ======================
def generate_report(chunks):
    total = len(chunks)
    valid = 0
    type_dist = defaultdict(int)
    lang_dist = defaultdict(int)

    for c in chunks:
        if MIN_TOKENS <= c["token_count"] <= MAX_TOKENS:
            valid += 1
        type_dist[c["type"]] += 1
        lang_dist[c["lang"]] += 1

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("=== 大疆文档处理报告 ===\n")
        f.write(f"总块数：{total}\n")
        f.write(f"有效块(200-512token)：{valid}\n")
        f.write(f"有效率：{round(valid/total, 2) if total else 0}\n\n")
        f.write("类型分布：\n")
        for k, v in type_dist.items():
            f.write(f"{k}: {v}\n")
        f.write("\n语言分布：\n")
        for k, v in lang_dist.items():
            f.write(f"{k}: {v}\n")

# ====================== 输出JSONL ======================
def save_jsonl(chunks):
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")

# ====================== 主入口 ======================
if __name__ == "__main__":
    chunks = batch_process()
    generate_report(chunks)
    save_jsonl(chunks)
    print("✅ 全部处理完成！")