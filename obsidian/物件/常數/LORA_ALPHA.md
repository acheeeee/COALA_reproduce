---
tags: [COALA, 物件, 常數, LoRA]
類別: 常數
型別: int
---
# LORA_ALPHA

← [[物件 總覽]]

**存什麼**：LoRA 的縮放係數，實際更新量 = `(alpha / r) · B·A`。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `32`（→ `alpha / r = 32 / 16 = 2`） |
| 定義於 | [[lora_setup.py]] |

維持 `alpha / r` 比值固定，之後改 [[LORA_RANK]] 時不需要重調 lr。

## 用在哪裡
- [[make_lora_config()]]
