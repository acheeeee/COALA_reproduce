---
tags: [COALA, 函式, 模型層, Stage2]
種類: 方法
檔案: projector.py
---
# DiscriminativeProjector.__init__()

← [[DiscriminativeProjector]] · [[projector.py]] · [[函式 總覽]]

```python
def __init__(self, d=D, d_prime=D_PRIME):
    super().__init__()
    self.mlp = nn.Sequential(nn.Linear(d, d_prime), nn.ReLU(), nn.Linear(d_prime, 1))
```

| 參數 | 預設 | 論文符號 |
|---|---|---|
| `d` | [[D]] = 576 | D |
| `d_prime` | [[D_PRIME]] = 576 | D′ |

對應論文 Eq. 2：
$$f(h_{ij}) = W_2 \cdot \text{ReLU}(W_1 h_{ij} + b_1) + b_2$$

| 論文 | 程式 | shape |
|---|---|---|
| W₁, b₁ | `mlp[0]` | `(576, 576)`、`(576,)` |
| ReLU | `mlp[1]` | — |
| W₂, b₂ | `mlp[2]` | `(1, 576)`、`(1,)` |

`nn.Sequential`：把幾個層串起來，輸入依序通過。
