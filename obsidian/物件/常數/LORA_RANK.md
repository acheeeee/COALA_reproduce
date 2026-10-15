---
tags: [COALA, 物件, 常數, LoRA]
類別: 常數
型別: int
---
# LORA_RANK

← [[物件 總覽]]

**存什麼**：LoRA 的秩 r（低秩矩陣 A、B 的中間維度）。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `16` |
| 定義於 | [[lora_setup.py]] |

## 怎麼決定的（論文沒給）
用 5.7M 可訓練預算反推：

```
每層：q_proj 576→576: 16×576 + 576×16 = 18,432
      v_proj 576→192: 16×576 + 192×16 = 12,288
30 層 × 30,720 = 921,600   ← 實測 trainable params 完全一致
```

## 用在哪裡
- [[make_lora_config()]] → [[lora_config]]

搭配 [[LORA_ALPHA]] = 32（`alpha/r = 2`）、[[LORA_DROPOUT]]、[[TARGET_MODULES]]。
