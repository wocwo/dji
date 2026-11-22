# 架构演进

## 当前

模块化单体：

```text
FastAPI
  → AgentRuntime（自研 ReAct）
  → ToolRegistry
  → JsonlKnowledgeBase（BM25 + rerank）
  → DeepSeek client
  → Grounding / ClaimVerifier / FactCoverage
```

## 稳定边界

`agent/ports.py` 定义：

- `LLMClientPort`
- `RetrieverPort`
- `RerankerPort`
- `VerifierPort`
- `MemoryPort`

现有实现先作为 adapter 使用，不改变运行逻辑。

## 迁移顺序

1. **Ports 落地**：已完成，`agent/ports.py`。
2. **LangGraph 编排层**：已实现真实节点循环 `agent/graph_runtime.py`，默认 `legacy`；
   - 节点：`prepare → think → act → think → ... → finalize`
   - `think` 负责 LLM/解析/证据闸门/事实覆盖，`act` 执行工具，`finalize` 做引用与校验。
3. **Feature flag**：
   ```text
   AGENT_ENGINE=legacy | graph
   ```
4. **冻结盲测 A/B**（完整 24 条，2026-11-21）：
   - legacy：24/24，平均 9.64s，426,772 token
   - graph：24/24，平均 11.45s，431,620 token（+1.1%）
   - 准确率无退化，但图执行略慢；因此保持 `legacy` 默认。
5. 下一步优化图节点的 token/延迟，再考虑默认切换。

## 不做什么

- 不把页面级检索重写成通用 chunk retriever；
- 不复制一套 prompt / 工具逻辑；
- 不让框架成为唯一事实来源。
