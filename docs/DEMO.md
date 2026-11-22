# 可复现 Demo

## 最快路径（本机）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
copy .env.agent.example .env.agent
# 填 DEEPSEEK_API_KEY；不填也可启动离线降级
python -m agent serve --host 127.0.0.1 --port 8077 --config configs/agent.deepseek.yaml
```

打开：

```text
http://127.0.0.1:8077/remote
```

健康检查：

```powershell
# 如果启用了 AGENT_ACCESS_TOKEN
$env:AGENT_ACCESS_TOKEN=(Get-Content .agent\access_token.txt -Raw).Trim()
python scripts\smoke_demo.py --base-url http://127.0.0.1:8077
```

Docker 演示默认密码是 `demo-token`：

```powershell
$env:AGENT_ACCESS_TOKEN="demo-token"
python scripts\smoke_demo.py --base-url http://127.0.0.1:8077
```

冒烟脚本会验证：

1. `/healthz` 返回知识库规模；
2. 异步任务能完成；
3. 回答非空；
4. 至少有一条页面级引用。

## Docker

```powershell
copy .env.agent.example .env.agent
docker compose up --build
```

镜像只装最小依赖，不安装 torch / sentence-transformers；JSONL 页面级 BM25 直接启动。

## 为什么能复现

- 语料固定在 `data/agent_manual_kb.jsonl`，不依赖 Chroma 构建；
- `AGENT_KB_BACKEND=jsonl` 显式指定；
- 模型配置在 `configs/agent.deepseek.yaml`；
- 无 Key 时服务自动降级离线策略，仍能演示检索/引用链路；
- 视觉是可选链路，没有 Key 时自动回退 OCR；
- 评测脚本固定用例，输出 JSON + Markdown。

## 已知限制

- 完整 DJI 手册数据受版权保护，开源时只应放演示语料；
- quick tunnel 会过期，不作为稳定部署；
- 视觉模型依赖 DashScope 网关，换网关需要改 `QWEN_BASE_URL`；
- 图片在浏览器端预览，不进服务器存储，刷新后历史不保留图片。
