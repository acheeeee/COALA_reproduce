---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# CTC_BLANK_ID

← [[物件 總覽]]

**存什麼**：CTC 的 blank 符號 id。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `0` |
| 定義於 | [[dataset.py]] |

## 用在哪裡
- 概念上被 [[CTC_CHAR2ID]] 保留（id 從 1 開始編）
- [[ctc_compress()]] 與 [[compute_ctc_loss()]] 的參數 `blank_id=0` —— **這兩個函式是直接寫死 `0`，沒有 import 這個常數**。值一致所以沒有 bug，但改其中一邊時要記得同步。
