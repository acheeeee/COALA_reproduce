---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (N, L)
dtype: float32
aliases: [mask]
---
# mask (projector)

← [[物件 總覽]]

**存什麼**：每個 entity 的哪些 token 位置是真的（1）、哪些是 pad（0）。

| 項目 | 內容 |
|---|---|
| shape | `(N, L)` |
| dtype | `float32`（1.0 / 0.0），可以直接和 [[f (token 分數)]] 相乘 |
| 定義於 | [[DiscriminativeProjector.forward()]] 的參數 |

## 怎麼造（[[projector.main()]]）
```python
mask = (torch.arange(L)[None, :] < lens[:, None]).float()
#        (1, L)                     (N, 1)   → broadcast → (N, L)
```
例：`lens = [2, 4, 1]`、L = 4 →
```
[[1, 1, 0, 0],
 [1, 1, 1, 1],
 [1, 0, 0, 0]]
```

## 為什麼一定要有
直接 `.mean(dim=1)` 會把 pad 也平均進去：分母變成 L 而不是 Lᵢ，而且 pad 位置的 f(h) **不是 0**（LM 輸出非零 + bias b₂）。

由 [[lens (entity 長度)]] 產生。
