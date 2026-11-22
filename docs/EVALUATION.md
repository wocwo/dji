# 评测与坏例

## 1. 端到端 holdout

固定 24 条 DJI 手册问题，覆盖 Mavic 4 Pro、Mini 4/5、Air 3S、Neo 2、Matrice 30、RC 2、Goggles 3、Matrice 350。

```powershell
python scripts\run_eval.py `
  --cases data\eval_holdout_v2.jsonl `
  --config configs\agent.deepseek.yaml `
  --output reports\eval_holdout_2026-11-22.json
```

| 指标 | 结果 |
|------|------|
| 通过 | **23 / 24（95.8%）** |
| 引用命中 | **24 / 24（100%）** |
| claim 覆盖 | **23 / 24（95.8%）** |
| 数字覆盖 | **23 / 24（95.8%）** |
| 平均耗时 | **10.3s** |
| 总耗时 | 247.2s |
| 最快 / 最慢 | 3.6s / 18.8s |

判定口径：

- `status=ok`；
- 至少一条 citation 的 `doc_id/page` 命中期望文档/页；
- 回答包含全部期望 claim；
- 回答包含全部期望数字。

## 2. 唯一失败：字符串归一化误判

失败用例：`h-air3s-charging`

```text
问：Air 3S 的充电管家能给几块电池充电？
期望：充电管家 / 三块 / USB-C
回答：DJI Air 3S 的充电管家最多可为 3 块智能飞行电池充电 [S2]
```

模型答案在语义上正确，但判定器只做字面匹配，`三块` ≠ `3 块`，因此 claim/number 判失败。

这不是模型答错，而是**评测器归一化不足**。下一版应：

- 中文数字 ↔ 阿拉伯数字归一化；
- 单位/空格/全半角归一化；
- 同义词表（夜间/光线不足）。

保留这条记录，是为了避免把“评测器 bug”包装成“模型 100%”。

## 3. 视觉评测

```powershell
python scripts\eval_vision.py --output reports\vision_eval_2026-11-22.json
```

用例：

| 用例 | 目的 | 结果 | 耗时 |
|------|------|------|------|
| `error_screenshot` | 报错文字 | PASS | 2.9s |
| `mavic3_product` | 正常产品图，不应幻觉损伤 | PASS | 2.5s |
| `blank_image` | 空白图，应拒答/不编造 | PASS | 1.4s |

3/3 通过；`visible_damage` 在正常图和空白图上都为空。

**但这不等于损伤识别达标**：没有真实坠机损伤集，不能对外宣称“能判断炸机损伤”。

## 4. 坏例清单

见 [BAD_CASES.md](BAD_CASES.md)。

## 5. 评测局限

- 24 条不是统计显著 benchmark，只是回归集；
- DeepSeek 输出有随机性，重跑可能 22–24；
- 视觉评测只有 3 张，缺少真实损伤、逆光、模糊、多损伤图；
- 没有人工双盲评分；
- 没有统计 token 成本 / 每类问题分层指标。

## 6. 下一步

1. 修中文数字/同义词归一化，重跑得到可信 24/24 或真实失败；
2. 扩到 50–100 条，按机型/任务分层；
3. 收集 20–30 张真实图片，标注可见损伤和不确定项；
4. 加 p50/p95 延迟和每千次成本；
5. 把评测接入 CI，防止回归。
