# FAISS 安装指南

## 方法1: Conda (推荐，支持GPU)

```bash
# 创建新环境
conda create -n faiss_env python=3.10
conda activate faiss_env

# 安装GPU版
conda install -c pytorch faiss-gpu

# 或CPU版
conda install -c pytorch faiss-cpu
```

## 方法2: Pip (仅CPU)

```bash
pip install faiss-cpu
```

**注意**: `faiss-gpu` 通常不能用pip安装，需要用conda。

## 验证安装

```python
import faiss
print(faiss.__version__)

# 检查GPU
try:
    import torch
    print(f"CUDA可用: {torch.cuda.is_available()}")
except:
    pass
```

## 如果都装不上

使用 `build_chroma_optimized.py` (纯Python，无需额外安装):

```bash
python build_chroma_optimized.py 1670000
```
