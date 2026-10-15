---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (L-1,)
dtype: bool
---
# change

← [[物件 總覽]]

**存什麼**：每個位置的預測和「前一個位置」是否不同（從第 2 個位置開始比）。

| 項目 | 內容 |
|---|---|
| shape | `(L-1,)` ← ⚠️ 比 [[pred]] 少一個 |
| dtype | `torch.bool` |
| 定義於 | [[ctc_compress()]] |

```python
change = pred[1:] != pred[:-1]
```

接上例：
```
pred[1:]  = [10,10, 0, 0, 7, 7, 7, 0,14]
pred[:-1] = [0,10,10, 0, 0, 7, 7, 7, 0 ]
change    = [T, F, T, F, T, F, F, T, T ]
```

長度少 1 → 前面補一個 `True`（`first`）變成 [[changed]]。
