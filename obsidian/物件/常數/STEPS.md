---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# STEPS

← [[物件 總覽]]

**存什麼**：過擬合測試要在同一個 batch 上訓練幾步。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `200` |
| 定義於 | [[overfit_test.py]] |

## 用在哪裡
- [[overfit_test.main()]]：`for step in range(1, STEPS + 1)`

不要和 [[STEPS_PER_EPOCH]]（[[dump_tb.py]]）搞混。
