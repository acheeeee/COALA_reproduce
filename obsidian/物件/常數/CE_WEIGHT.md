---
tags: [COALA, 物件, 常數, loss]
類別: 常數
型別: float
---
# CE_WEIGHT

← [[物件 總覽]]

**存什麼**：Stage 1 總 loss 裡 CE（LM 轉寫）loss 的權重。

| 項目 | 內容 |
|---|---|
| 型別 | `float` |
| 值 | `1.0` |
| 定義於 | [[model.py]] |

## 用在哪裡
- [[COALAModel.forward_stage1()]]，搭配 [[CTC_WEIGHT]]
