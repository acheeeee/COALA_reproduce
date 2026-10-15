---
tags: [COALA, 物件, 常數, loss]
類別: 常數
型別: float
---
# CTC_WEIGHT

← [[物件 總覽]]

**存什麼**：Stage 1 總 loss 裡 CTC loss 的權重。

| 項目 | 內容 |
|---|---|
| 型別 | `float` |
| 值 | `0.3`（論文 3.4 有給） |
| 定義於 | [[model.py]] |

## 用在哪裡
- [[COALAModel.forward_stage1()]]：`total = CTC_WEIGHT * ctc_loss + CE_WEIGHT * ce_loss`

→ [[loss (total)]] = 0.3 × [[ctc_loss]] + 1.0 × [[ce_loss]]
