---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: dict
aliases: [prompt]
---
# prompt (生成 prompt)

← [[物件 總覽]]

**存什麼**：生成轉寫時給 LM 的 prompt —— 只到 assistant turn 開頭，沒有答案。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[build_gen_prompt()]] |
| 用在 | [[transcribe_one()]] → `model.lm.generate(...)` |

| 鍵 | shape |
|---|---|
| `inputs_embeds` | `(1, n1 + L' + n2, 576)` |
| `attention_mask` | `(1, n1 + L' + n2)`，全 1 |

## 和訓練版的差別
```
訓練：[seg1][audio][seg2][target]  → teacher forcing 算 loss
生成：[seg1][audio][seg2]          → LM 自己接著寫
```
已經有 batch 維（1），因為 `generate()` 要求。
