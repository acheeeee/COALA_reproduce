---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[int]
aliases: [n2]
---
# neg_idx

← [[物件 總覽]]

**存什麼**：[[full_list (E)]] 裡除了 [[pos_idx]] 以外的所有位置 = E⁻。

| 項目 | 內容 |
|---|---|
| 型別 | `list[int]` |
| 長度 | `len(full_list) - len(pos_idx)` |
| 由誰產生 | [[split_pos_neg()]] |

這句有 rare word 時，索引 0（`<unbiased>`）**在這裡面**。
