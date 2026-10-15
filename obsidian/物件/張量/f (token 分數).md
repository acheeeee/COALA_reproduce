---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (N, L)
dtype: float32
---
# f (token 分數)

← [[物件 總覽]]

**存什麼**：每個 entity 的每個 token 各自的匹配分數 —— 論文 Eq. 2 的 f(h_ij)。

| 項目 | 內容 |
|---|---|
| shape | MLP 輸出 `(N, L, 1)` → `.squeeze(-1)` → `(N, L)` |
| 定義於 | [[DiscriminativeProjector.forward()]] |

```python
f = self.mlp(h)          # (N, L, 576) → (N, L, 1)
f = f.squeeze(-1)        # → (N, L)
```

> [!tip] 為什麼是 `squeeze(-1)` 而不是 `squeeze()`
> 不指定維度會把**所有**大小為 1 的維度壓掉。N=1 或 L=1 時 shape 就錯了。

## 去處
`(f * mask).sum(dim=1) / mask.sum(dim=1)` → [[s (entity 分數)]]
