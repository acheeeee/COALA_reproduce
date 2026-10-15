---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (L,)
dtype: bool
aliases: [first]
---
# changed

← [[物件 總覽]]

**存什麼**：補齊長度後的 [[change]] —— 第 0 個位置一律算「有變化」（它前面沒有東西）。

| 項目 | 內容 |
|---|---|
| shape | `(L,)` |
| dtype | `torch.bool` |
| 定義於 | [[ctc_compress()]] |
| 零件 | `first = torch.ones(1, dtype=torch.bool, device=pred.device)` → `(1,)` 的 `[True]` |

```python
changed = torch.cat([first, change])
```

> [!danger] `first` 一定要指定 `device=pred.device`
> 不寫的話 `first` 在 CPU、`change` 在 GPU → `torch.cat` 報錯。CPU 上測試完全看不出來。
