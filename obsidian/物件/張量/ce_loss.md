---
tags: [COALA, 物件, 張量, loss]
類別: 張量
型別: torch.Tensor
shape: () 純量
dtype: float32
---
# ce_loss

← [[物件 總覽]]

**存什麼**：LM 預測逐字稿下一個 token 的 cross-entropy（只在 target 位置算）。

| 項目 | 內容 |
|---|---|
| shape | `()` |
| 定義於 | [[COALAModel.forward_stage1()]]：`ce_loss = out.loss` |
| 誰算的 | HF 的 LM 內部：給了 [[labels]] 就自動 shift 一位、略過 [[IGNORE_INDEX]] 位置、取平均 |

## 量級參考

| 狀態 | 值 | 意義 |
|---|---|---|
| 隨機權重的理論值 | ln(49152) = 10.80 | 49152 類均勻亂猜 |
| **剛載入預訓練權重** | **5.60** | ≪ 10.80 → 證明 SmolLM2 權重確實載進來了 |

## 去處
- 乘上 [[CE_WEIGHT]] = 1.0 加進 [[loss (total)]]

來自 [[out (LM 輸出物件)]] 的 `.loss`。
