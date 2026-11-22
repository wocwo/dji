"""
独立工具：从指定网页解析 PDF 链接并下载到 data/dji_manuals。

与 RAG 核心流水线无 import 关系；用于采集手册源文件。
"""

import requests
from bs4 import BeautifulSoup
import os
import time

# 配置项
TARGET_URL = "https://www.romo.tech/cn/romo/downloads#doc"  # 替换成你的目标网页
SAVE_DIR = "data/dji_manuals"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 10  # 请求超时时间（秒）

# 创建保存文件夹
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"✅ 保存文件夹已准备好: {os.path.abspath(SAVE_DIR)}")
except Exception as e:
    print(f"❌ 创建文件夹失败: {e}")
    exit(1)

# 配置请求头（模拟浏览器，避免反爬）
headers = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
}

# 第一步：获取网页内容
try:
    print(f"\n🔍 正在请求网页: {TARGET_URL}")
    response = requests.get(TARGET_URL, headers=headers, timeout=TIMEOUT)
    response.raise_for_status()  # 触发HTTP错误（404/403等）
    print(f"✅ 网页请求成功，状态码: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"❌ 网页请求失败: {e}")
    exit(1)

# 第二步：解析所有链接（不仅限于a标签，还包括iframe/embed）
soup = BeautifulSoup(response.text, "html.parser")
all_links = []

# 1. 提取a标签的链接
a_links = soup.find_all("a", href=True)
for link in a_links:
    all_links.append(link["href"])

# 2. 提取iframe/embed中的PDF链接（在线预览的PDF）
iframe_links = soup.find_all("iframe", src=True)
for link in iframe_links:
    if "pdf" in link["src"].lower():
        all_links.append(link["src"])

embed_links = soup.find_all("embed", src=True)
for link in embed_links:
    if "pdf" in link["src"].lower():
        all_links.append(link["src"])

# 去重并筛选可能的PDF链接
unique_links = list(set(all_links))
pdf_candidates = [link for link in unique_links if "pdf" in link.lower()]

print(f"\n📊 解析结果：")
print(f"- 网页中找到所有链接数: {len(unique_links)}")
print(f"- 疑似PDF链接数: {len(pdf_candidates)}")
if pdf_candidates:
    print(f"- 疑似PDF链接列表: {pdf_candidates}")
else:
    print(f"⚠️  未找到任何疑似PDF链接，检查网页是否有PDF或链接格式是否特殊")
    exit(0)

# 第三步：下载PDF
downloaded_count = 0
for idx, href in enumerate(pdf_candidates, 1):
    # 处理相对链接（更健壮的拼接方式）
    try:
        from urllib.parse import urljoin
        pdf_url = urljoin(TARGET_URL, href)  # 自动拼接绝对URL
        print(f"\n[{idx}/{len(pdf_candidates)}] 正在处理: {pdf_url}")

        # 测试PDF链接是否可访问
        pdf_response = requests.get(pdf_url, headers=headers, timeout=TIMEOUT, stream=True)
        pdf_response.raise_for_status()

        # 验证响应是否为PDF（通过Content-Type）
        content_type = pdf_response.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower():
            print(f"⚠️  链接不是PDF文件，Content-Type: {content_type}")
            continue

        # 提取文件名（优先从响应头获取，否则用URL最后部分）
        filename = None
        content_disposition = pdf_response.headers.get("Content-Disposition", "")
        if "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"\'')
        if not filename:
            filename = os.path.basename(pdf_url).split("?")[0]  # 去掉URL参数
            if not filename.endswith(".pdf"):
                filename += ".pdf"  # 补全PDF后缀

        # 保存文件
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, "wb") as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)

        downloaded_count += 1
        print(f"✅ 下载成功: {filename} (保存路径: {os.path.abspath(filepath)})")
        time.sleep(1)  # 延迟1秒，避免请求过快被反爬

    except requests.exceptions.RequestException as e:
        print(f"❌ 下载失败: {href} → 错误: {e}")
    except Exception as e:
        print(f"❌ 处理链接时出错: {href} → 错误: {e}")

# 最终统计
print(f"\n🎉 下载完成！")
print(f"- 总计处理疑似PDF链接: {len(pdf_candidates)}")
print(f"- 成功下载: {downloaded_count}")
print(f"- 保存位置: {os.path.abspath(SAVE_DIR)}")