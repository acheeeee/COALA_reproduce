---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (N,)
dtype: int64
---
# lens (entity 長度)

← [[物件 總覽]]

**存什麼**：每個候選 entity 實際有幾個 token —— 論文 Eq. 1 的 Lᵢ。

| 項目 | 內容 |
|---|---|
| shape | `(N,)` |
| 定義於 | [[projector.main()]]：`torch.randint(1, L + 1, (N,))`（測試用的假長度） |

正式 Stage 2 會由 [[tokenizer]] 切每個 entity 的結果得到。

→ 用來產生 [[mask (projector)]]
