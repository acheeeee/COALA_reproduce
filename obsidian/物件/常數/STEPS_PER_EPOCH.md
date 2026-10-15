---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# STEPS_PER_EPOCH

← [[物件 總覽]]

**存什麼**：train-clean-100、batch 4、`drop_last=True` 時一個 epoch 有幾步。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `7134`（= 28,536 句 / 4） |
| 定義於 | [[dump_tb.py]] |

## 用在哪裡
- [[dump_tb.main()]]：`--steps-per-epoch` 的預設值
- [[epoch_of()]]：`(step - 1) // steps_per_epoch` 把 step 換算成 epoch

換資料集或 batch size 後這個數字就不對了，要用命令列參數覆寫。
