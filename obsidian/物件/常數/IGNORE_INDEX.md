---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# IGNORE_INDEX

← [[物件 總覽]]

**存什麼**：[[labels]] 裡「這個位置不算 loss」的標記值。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `-100` |
| 定義於 | [[prompt.py]] |

## 為什麼是 -100
PyTorch `CrossEntropyLoss` 的 `ignore_index` 預設就是 `-100`，HF 的 LM 內部算 loss 時沿用它。所以 [[labels]] 填 -100 的位置會被自動略過，不用自己寫 mask。

## 用在哪裡
- [[build_stage1_sample()]]：seg1 + audio + seg2 全填 -100，只有 target 放真 token id
- [[pad_batch()]]：pad 位置的 labels 也填 -100
- [[prompt.main()]]：`(labels != IGNORE_INDEX).sum()` 數出算 loss 的位置數
