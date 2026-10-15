---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[str]
aliases: [full_list, full, full2, E]
---
# full_list (E)

← [[物件 總覽]]

**存什麼**：論文的完整 biasing list E = {e₀, e₁, …, e_M}，e₀ 是 `<unbiased>`。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]` |
| 長度 | M + 1 |
| 長相 | `["<unbiased>", "alpha", "intermingled", "mated", ...]` |
| 由誰產生 | [[add_unbiased()]]：`[UNBIASED] + list(biasing_list)` |

## 去處
- [[split_pos_neg()]] → [[pos_idx]]、[[neg_idx]]
- Stage 2（尚未實作）：每個元素各跑一次 LM forward → 121 個 [[s (entity 分數)]]
