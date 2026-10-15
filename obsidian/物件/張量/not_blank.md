---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (L,)
dtype: bool
---
# not_blank

← [[物件 總覽]]

**存什麼**：每個位置的預測是不是「真字元」（不是 blank）。

| 項目 | 內容 |
|---|---|
| shape | `(L,)` |
| dtype | `torch.bool` |
| 定義於 | [[ctc_compress()]] |

```python
not_blank = pred != blank_id
```

接上 [[pred]] 的例子：
```
pred      = [0,10,10, 0, 0, 7, 7, 7, 0,14]
not_blank = [F, T, T, F, F, T, T, T, F, T ]
```

和 [[changed]] 做 AND → [[keep]]
