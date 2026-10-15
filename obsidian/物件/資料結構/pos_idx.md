---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[int]
aliases: [p2]
---
# pos_idx

← [[物件 總覽]]

**存什麼**：[[full_list (E)]] 裡哪些位置是正樣本 E⁺。

| 項目 | 內容 |
|---|---|
| 型別 | `list[int]` |
| 由誰產生 | [[split_pos_neg()]] |

## 兩種情況

| 這句有 rare word？ | pos_idx |
|---|---|
| 有 | 那些 rare word 在 E 裡的索引，例 `[2, 3]` |
| 沒有 | `[0]`（`<unbiased>` 的位置） |

**永遠不會是空的** → 三個 loss 的分母 |E⁺| 永遠 ≥ 1。

## 之後的去處
`losses.py`（尚未實作）會把它轉成 bool 遮罩 `pos`，搭配 [[s (entity 分數)]] 算 loss。
