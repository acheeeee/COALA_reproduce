---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (B,)
dtype: int64
---
# input_lengths

← [[物件 總覽]]

**存什麼**：每句的 CTC 輸入長度（有幾個位置）。

| 項目 | 內容 |
|---|---|
| shape | `(B,)` |
| 值 | **全部等於 T**（750，stack=2） |
| 定義於 | [[compute_ctc_loss()]] |

```python
input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
```

## 為什麼全部一樣
Whisper 把每句都 pad 成 30 秒 → 每句都有 T 個位置，沒有「輸入端的 padding」。（短句後半段是靜音，CTC 會學著輸出 blank。）

搭配 [[ctc_lens]]（目標端長度）一起交給 `F.ctc_loss`。
