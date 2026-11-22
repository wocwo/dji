# 简历项目描述

## 项目名称

**DJI 手册 Agent：带页面级引用的多模态维修问答系统**

## 一句话

把 26,596 页 DJI 官方手册做成可审计的 Agent 检索系统：页面级引用、ReAct 工具调用、多轮记忆、图片观察和用户隔离，并用固定评测集和坏例清单约束幻觉。

## 技术栈

`Python 3.11` · `FastAPI` · `SQLite` · `BM25 / JSONL` · `ONNX bge-reranker` · `DeepSeek` · `Qwen3-VL` · `RapidOCR` · `pytest`

## 简历 Bullet（可直接用）

- **构建 DJI 手册 Agent 检索系统**，覆盖 **26,596 页 / 1,067 份文档**；采用页面级 BM25 + 规则重排 + ONNX reranker，回答携带 `[S1]` 文档/页码引用，支持 Mavic、Mini、Air、Neo、Matrice、RC、Goggles 多机型。
- **实现真 ReAct 工具循环**（检索、取页、计算、笔记、计划），配合 Evidence Gate、Grounding、Claim Verifier 和 Fact Coverage，降低无证据作答；手册未覆盖时明确输出「通用建议（非手册）」并追问。
- **实现长上下文记忆**：SQLite 持久化多轮 turn、滚动摘要、历史相关轮次检索；按窗口增量更新摘要，避免历史线性膨胀。
- **实现 fast / normal / auto 三种模式**：fast 约 4k token、5–8s；normal 约 9k token、8–15s；auto 按电池/固件/返航等高风险词自动路由。
- **接入 Qwen3-VL 图片观察**：RapidOCR 读文字 + Qwen3-VL 输出机型/指示灯/可见损伤/置信度；图片观察与手册事实分层，不把视觉猜测写成页码；通过 base64 data URI 替代 OSS 临时上传，将视觉调用从 43–50s 降至约 2–3s（直连）。
- **建立评测与坏例机制**：24 条 holdout 端到端评测 **23/24（95.8%）**、引用命中 **24/24**、平均 **10.3s**；视觉负样本 3/3 通过；保留“中文数字导致误判”的坏例，避免把评测器 bug 包装成模型 100%。
- **完成安全与多用户隔离**：Basic 用户名映射、session 命名空间 `u:<user>:<id>`、job owner 校验、HttpOnly cookie、移除 URL token、按用户限流；177 个自动化测试通过。
- **交付可复现 Demo**：最小依赖 Dockerfile / Compose、JSONL 演示语料、`smoke_demo.py` 冒烟脚本、固定评测脚本与设计文档。

## 可讲的技术难点

1. 为什么手册问答不能只做向量 top-k：精确参数和页码证据。
2. Evidence Gate / Grounding 如何减少“看起来对”的幻觉。
3. 滚动摘要 + 历史检索如何控制 token。
4. 视觉模型如何只做 observation，不冒充手册。
5. base64 data URI 为什么比 SDK 本地文件快一个数量级。
6. 多用户隔离为什么要做 session 命名空间和 job owner。

## 诚实边界

- 23/24 不是统计显著 benchmark，重跑可能有波动；
- 视觉只测了 3 张图，没有真实坠机损伤集；
- `qwen-vl` 在当前网关不可用，实际使用 `qwen3-vl-plus`；
- 单进程 SQLite 隔离不是生产多租户；
- 手册数据有版权，公开仓库只放演示语料。

## 30 秒口述版本

> 我做了一个 DJI 手册 Agent，不是简单 RAG。它把 2.6 万页手册做成页面级证据链：Agent 自己决定搜什么、取哪页，回答必须带页码引用；手册没覆盖就明确说没有。后面加了长上下文、图片观察和用户隔离。评测是 24 条 holdout 23 条通过、引用 24/24；视觉 3 个负样本没有幻觉损伤。我也保留了失败案例，因为维修场景最重要的是知道哪里不可靠。
