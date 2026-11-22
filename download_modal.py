"""
一次性工具：通过 HF 镜像下载 SentenceTransformer 到 ./mini_model（文件名沿用历史拼写）。

运行前无需改文件名；仅用于准备本地嵌入模型。
"""

import os
# 强制用国内镜像，解决连接失败
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer

# 下载模型到本地
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model.save("./mini_model")  # 存在你电脑里
print("✅ 模型下载完成！已保存到 mini_model 文件夹")