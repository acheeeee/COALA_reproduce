---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (T, B, C)
dtype: float32
---
# log_probs

← [[物件 總覽]]

**存什麼**：CTC 各類別的 log 機率，維度順序換成 `F.ctc_loss` 要的格式。

| 項目 | 內容 |
|---|---|
| shape | `(T, B, C)` = `(750, B, 29)`（stack=2） |
| 值域 | 全部 ≤ 0（log 機率） |
| 定義於 | [[compute_ctc_loss()]] |

```python
log_probs = logits.log_softmax(dim=-1).transpose(0, 1)   # (B,T,C) → (T,B,C)
```

> [!warning] 兩個不會報錯的坑
> 1. 忘了 `log_softmax` → 照樣能算，只是數字錯。檢查法：log 機率**永遠是負的**
> 2. 忘了 `transpose` → B 和 T 對調，shape 剛好吻合時也不會報錯

由 [[ctc_logits]] 而來，送進 `F.ctc_loss` 算出 [[ctc_loss]]。
