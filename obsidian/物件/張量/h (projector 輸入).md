---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (N, L, 576)
dtype: float32
aliases: [h]
---
# h (projector 輸入)

← [[物件 總覽]]

**存什麼**：Stage 2 中，每個候選 entity 的每個 token 在 LM **最後一層**的 hidden state —— 論文 Eq. 1 的 h_ij。

| 項目 | 內容 |
|---|---|
| shape | `(N, L, 576)` |
| N | 候選數 = M + 1 = **121**（含 [[UNBIASED]]） |
| L | 最長 entity 的 token 數（短的 pad） |
| 576 | [[D]] |
| 定義於 | [[DiscriminativeProjector.forward()]] 的參數 |

> 和 [[h (adapter 輸出)]] 名字一樣但完全不同：那個是音訊進 LM **之前**，這個是 LM **輸出**。

## 從哪來（Stage 2，尚未實作）
`lm(..., output_hidden_states=True).hidden_states[-1]`，取出 entity token 對應的那幾個位置。

## 去處
[[DiscriminativeProjector]] → [[f (token 分數)]] → [[s (entity 分數)]]

目前只在 [[projector.main()]] 用 `torch.randn(121, 7, 576)` 假資料測試。
