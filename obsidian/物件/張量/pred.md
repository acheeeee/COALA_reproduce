---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (L,)
dtype: int64
---
# pred

← [[物件 總覽]]

**存什麼**：單句每個位置 CTC 預測的字元 id（取最大分數那一類）。

| 項目 | 內容 |
|---|---|
| shape | `(L,)`，L = 750（stack=2） |
| dtype | `torch.long`，值域 0（blank）~ 28 |
| 定義於 | [[ctc_compress()]] |

```python
pred = logits.argmax(dim=-1)     # (L, 29) → (L,)
```

## 例子（概念）
```
pred      = [0,10,10, 0, 0, 7, 7, 7, 0,14]
             ─  h  h  ─  ─  e  e  e  ─  l     （h=10, e=7, l=14，0=blank）
```

## 衍生
[[not_blank]]、[[change]] → [[changed]] → [[keep]]
