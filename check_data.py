"""
从 dji_chunks_final.jsonl 抽样校验文本质量，并写出 sample_chunks.json（需本地存在 jsonl）。
"""

import json

def find_valid_chunks():
    f = open('dji_chunks_final.jsonl', 'r', encoding='utf-8')
    valid_chunks = []
    for i, line in enumerate(f):
        if i > 50000:
            break
        chunk = json.loads(line)
        text = chunk.get('text', '')
        if len(text) > 100 and '\u0000' not in text:
            # Check if has Chinese characters
            has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
            if has_chinese:
                valid_chunks.append({
                    'index': i,
                    'text': text[:500],
                    'model': chunk.get('model', ''),
                    'type': chunk.get('type', ''),
                    'keywords': chunk.get('keywords', []),
                    'section': chunk.get('section', '')
                })
                if len(valid_chunks) >= 5:
                    break
    f.close()
    
    # Write to file
    with open('sample_chunks.json', 'w', encoding='utf-8') as out:
        json.dump(valid_chunks, out, ensure_ascii=False, indent=2)
    print(f"Found {len(valid_chunks)} valid chunks")

if __name__ == "__main__":
    find_valid_chunks()
