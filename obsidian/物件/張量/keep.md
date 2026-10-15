---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (L,)
dtype: bool
---
# keep

← [[物件 總覽]]

**存什麼**：哪些位置要保留給 LM —— 「不是 blank」**而且**「和前一個不同」。

| 項目 | 內容 |
|---|---|
| shape | `(L,)` |
| dtype | `torch.bool` |
| 定義於 | [[ctc_compress()]] |

```python
keep = not_blank & changed
return hidden[keep]          # 布林遮罩取列 → (L', 576)
```

接上例：
```
pred      = [0,10,10, 0, 0, 7, 7, 7, 0,14]
not_blank = [F, T, T, F, F, T, T, T, F, T ]
changed   = [T, T, F, T, F, T, F, F, T, T ]
keep      = [F, T, F, F, F, T, F, F, F, T ]   → 留 3 個位置：h, e, l
```

`keep.sum()` = L' = [[audio_tok]] 的長度。
