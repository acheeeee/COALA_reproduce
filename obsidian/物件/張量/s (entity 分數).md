---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (N,)
dtype: float32
aliases: [s, avg, s_ref]
---
# s (entity 分數)

← [[物件 總覽]]

**存什麼**：每個候選 entity 一個純量分數 sᵢ —— 「這個詞出現在這段音訊裡」的匹配強度（論文 Eq. 1）。

$$s_i = \frac{1}{L_i}\sum_{j=1}^{L_i} f(h_{ij})$$

| 項目 | 內容 |
|---|---|
| shape | `(N,)`，N = 121 |
| 定義於 | [[DiscriminativeProjector.forward()]] 回傳（區域變數名 `avg`） |
| 中間量 | `total = (f * mask).sum(dim=1)`、`count = mask.sum(dim=1)`，都是 `(N,)` |
| 別名 | [[projector.main()]] 裡的 `s`、`s_ref`（逐個計算的對照組） |

## 長相（論文圖 2b 的例子）

| i | entity | sᵢ |
|---|---|---|
| 0 | `<unbiased>` | −2.0 |
| 1 | Watson | 3.0 |
| 2 | solved | 1.0 |
| 3 | survey | −1.0 |
| 4 | collect | −3.0 |

## 之後的去處（尚未實作）
- `losses.py` 的三個 loss：搭配 [[pos_idx]] / [[neg_idx]]（E⁺ / E⁻）
- BTI：排序取 top-K，並用 s₀（`<unbiased>`）當門檻
