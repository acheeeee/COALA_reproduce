---
tags: [COALA, 函式, 模型層, Stage2]
種類: 方法
檔案: projector.py
---
# DiscriminativeProjector.forward()

← [[DiscriminativeProjector]] · [[projector.py]] · [[函式 總覽]]

**功能**：token 分數（Eq. 2）→ 只對有效 token 取平均（Eq. 1）。

```python
def forward(self, h, mask):
    f = self.mlp(h)              # (N, L, 576) → (N, L, 1)
    f = f.squeeze(-1)            # → (N, L)
    total = (f * mask).sum(dim=1)
    count = mask.sum(dim=1)
    avg = total / count
    return avg
```

## 輸入 → 輸出

| | shape | 物件 |
|---|---|---|
| 輸入 `h` | `(N, L, 576)` | [[h (projector 輸入)]] |
| 輸入 `mask` | `(N, L)`，1/0 | [[mask (projector)]] |
| 輸出 | `(N,)` | [[s (entity 分數)]] |

## 內部物件

| 物件 | shape | 角色 |
|---|---|---|
| [[f (token 分數)]] | `(N, L)` | 每個 token 的分數 |
| `total` | `(N,)` | 有效 token 分數的總和（pad 被乘 0） |
| `count` | `(N,)` | 有效 token 數 = Lᵢ |
| `avg` | `(N,)` | sᵢ |

## 已知的邊界情況

| 情況 | 後果 | 防法 |
|---|---|---|
| 某 entity 0 個有效 token | 除以 0 → NaN | `count.clamp(min=1)`（實際上每個 entity 至少 1 token，不會發生） |
| pad 位置的 h 是 NaN / inf | `NaN × 0 = NaN`，污染結果 | 改用 `f.masked_fill(mask == 0, 0)` |

## 被誰呼叫
[[projector.main()]]；之後由 Stage 2 的 forward 呼叫。
