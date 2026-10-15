---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (B, L, 29)
dtype: float32
aliases: [logits]
---
# ctc_logits

← [[物件 總覽]]

**存什麼**：每個音訊位置「是哪個字元」的未正規化分數（29 類 = blank + 28 字元）。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor` |
| shape | `(B, L, 29)`，L = 1500/stack（stack=2 → 750） |
| dtype | `float32` |
| 別名 | `logits`（[[ctc_compress()]] 收單句 `(L, 29)`、[[compute_ctc_loss()]] 收 batch） |

> 和 [[logits (LM 輸出)]]（49152 類）是不同東西。

## 兩個用途
1. **算 loss**：[[compute_ctc_loss()]] → `log_softmax` → [[log_probs]] → [[ctc_loss]]
2. **挑位置**：[[ctc_compress()]] → `argmax` → [[pred]] → [[keep]]

## 從哪來
- [[CTCHead.forward()]]（`nn.Linear(576, 29)`），在 [[COALAModel.encode_audio()]] 中呼叫
