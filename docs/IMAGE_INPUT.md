# 图片输入 / 视觉观察

## 当前链路

```text
浏览器选择图片
 → 本地缩略图预览（Object URL，不落盘）
 → POST /v1/attachments/analyze
     ├─ RapidOCR 读文字
     └─ Qwen3-VL 结构化观察（qwen3-vl-plus）
 → 合并为附件上下文
 → 问题 + 手册检索
 → 带 [S1] 手册引用 + 图片观察（非手册）
```

## 视觉模型

- `QWEN_VL_MODEL=qwen3-vl-plus`
- DashScope 原生 `MultiModalConversation`
- base64 data URI，不走 OSS 临时上传
- `qwen-vl` / `qwen-vl-plus` / `qwen-vl-max` 在当前 MaaS 网关 404，不可用
- `qwen-vl-ocr` 已按要求停用

## UI

- 上传后输入框上方显示缩略图、文件名、状态、移除按钮；
- 发送后图片显示在用户消息气泡里；
- 只发图片也可以，任务默认“请分析这张图片。”；
- 图片不存服务器，刷新后历史消息不恢复图片。

## 视觉输出

```json
{
  "model_guess": "DJI Mavic 3 Enterprise",
  "screen_text": "Battery Low",
  "leds": "前臂末端红色 LED",
  "visible_damage": {},
  "confidence": 0.95,
  "uncertain": []
}
```

## 边界

- 只描述可见内容，不推断内部损坏；
- 不诊断电池鼓包/云台排线断裂；
- 低置信度写入 `uncertain`，回答需要追问；
- 图片观察不能生成 `[S1]` 手册页码；
- 没有真实坠机损伤集，不能宣称损伤识别已达标；
- 第三方视觉 API 会收到图片，需要告知用户。
