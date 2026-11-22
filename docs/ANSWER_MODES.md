# 回答模式

## fast

- 预检索 1 次
- LLM 1 次
- top_k 3
- 关闭 claim / fact coverage
- 目标：5–7s，约 4k token

## normal

- 预检索 1 次
- 多步检索 + 回答
- top_k 6
- max_passage_chars 900
- max_observation_chars 5000
- 保留高风险 claim verifier / fact coverage
- 目标：8–12s，约 9k token

## auto

- 页面默认模式
- 含电池/充电/固件/返航/坠机/进水等高风险词 → normal
- 其他问题 → fast
- 实测：
  - 视觉系统问题：约 3.7k token，8s
  - 电池低温问题：约 8.1k token，10s

## 2026-11-21 A/B

24 条盲测：

| 模式 | 通过 | 平均耗时 | token |
|------|------|----------|-------|
| 旧 normal | 24/24 | 9.64s | 426,772 |
| 新 normal | 23/24 | 6.52s | 225,477 |

新 normal 唯一失败是 `h-m4p-vision`：回答覆盖了“光线不足/特别暗”，
但评测期望词是“夜间”，属于同义词差异，不是事实错误。

因此 normal 优化保留，但继续修正评测同义词，而不是为了单词改回答。
