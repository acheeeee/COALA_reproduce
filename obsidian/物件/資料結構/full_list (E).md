---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[str]
aliases: [full_list, full, full2, E, candidates, cands]
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
- [[build_stage2_sample()]] 的 `candidates` 參數：每個元素變成一條尾巴
- Stage 2（8b 尚未實作）：→ 121 個 [[s (entity 分數)]]
