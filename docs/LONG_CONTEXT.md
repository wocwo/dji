# 长上下文与会话记忆

## 当前结构

```text
用户问题
  + 最近 6 轮原文
  + 更早轮次的滚动摘要
  + 当前问题相关的历史轮次
  → LangGraph / ReAct
```

## SQLite

`AgentMemory` 现在维护：

| 表 | 用途 |
|----|------|
| `turns` | 每轮 user/assistant 原文 |
| `session_summaries` | 每个 session 的滚动摘要 + last_turn_id |
| `episodes` | 每次运行的答案、引用、steps |
| `notes` | 长期笔记 |

## 滚动摘要

`Agent._maybe_update_summary()`：

- 最近 `history_turns * 2` 条保留原文；
- 更早的轮次压缩成条目：
  ```text
  - 用户：...
  - 回答：...
  ```
- 只处理 `id > last_turn_id` 的新旧轮次，避免重复；
- 摘要上限约 1400 字符；
- 写入 `session_summaries`。

目前是提取式摘要，不额外调用 LLM，所以不增加 API token。

## 历史检索

`AgentMemory.search_turns()`：

- 同一 session 内按词面重叠排序；
- 排除最近轮次，避免重复；
- 取 top 3；
- 与摘要一起作为 `system` 上下文注入。

## 会话列表

- `GET /v1/sessions`
- `GET /v1/sessions/{id}`
- 远程页左侧“最近会话”可恢复历史；
- `新问题` 生成新 session。

## 限制

- 历史检索是词面匹配，不是向量检索；
- 摘要是提取式，不是语义摘要；
- 6 轮窗口之外的细节可能丢失；
- 目前单用户共享 SQLite，多人部署需要按 `user_id` 隔离。
