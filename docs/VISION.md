# Qwen-VL 云端视觉接入

## 1. 获取 Key

阿里云百炼 / DashScope 控制台创建 API Key，形如 `sk-...`。

写入 `.env.agent`（不要提交）：

```env
DASHSCOPE_API_KEY=sk-...
QWEN_MODE=native
QWEN_BASE_URL=https://cn-beijing.maas.qianwenaiapi.com/api/v1
QWEN_VL_MODEL=qwen3-vl-plus
```

这个 MaaS 网关实测：

- `qwen3-vl-plus`：唯一可用的非 OCR 视觉模型，输出结构化观察；
- `qwen-vl` / `qwen-vl-plus` / `qwen-vl-max`：404，不存在；
- `qwen-vl-ocr`：可用，但按要求不再启用。

## 传输优化

早期把本地图片写成临时文件交给 SDK，会触发 OSS 上传，公网接口 43–50s。现在改成 **base64 data URI** 直接传给 `MultiModalConversation`：

- 错误截图：约 2–3s
- 产品图结构化观察：直连约 2.5s；公网约 5–20s
- 不再产生临时上传文件

## 2. 接口

```text
POST /v1/attachments/analyze
multipart/form-data:
  file: 图片
  question: 可选问题
```

返回：

```json
{
  "ocr": {"text": "...", "lines": []},
  "vision": {
    "model_guess": "DJI Mini 4 Pro",
    "screen_text": "Battery Low",
    "leds": "LED1 闪烁",
    "visible_damage": {"battery": "疑似鼓包"},
    "confidence": 0.82,
    "uncertain": ["云台排线"],
    "error": ""
  },
  "vision_available": true
}
```

没有 Key 时：

```json
{"vision_error": "未配置 DASHSCOPE_API_KEY / QWEN_API_KEY", "vision_available": false}
```

OCR 仍然工作，前端自动回退。

## 3. 实现位置

- `agent/vision.py`：`QwenVLClient`、`VisionObservation`
  - `QWEN_MODE=native` 使用 DashScope `MultiModalConversation`
  - 模型名含 `ocr` 时按纯文本解析
- `agent/server.py`：`/v1/attachments/analyze`
- `web/remote.html`：上传后读取 `ocr + vision`

## 4. 边界

- 视觉只输出观察，不做维修诊断；
- 图片观察标注“非手册”，不能伪装成手册页码；
- 低置信度必须写进 `uncertain`；
- 图片压缩到 1600px、JPEG 86，不保留 EXIF；
- 无视觉 Key 时降级 OCR，不阻断回答。

## 5. 验证

```bash
curl -u dji:你的密码 \
  -F "file=@error.png" \
  -F "question=这张截图说明什么" \
  https://你的域名/v1/attachments/analyze
```

预期：

- `vision_available=true`
- `model_guess` / `screen_text` / `confidence` 有值
- 回答中能看到“视觉观察（非手册）”
