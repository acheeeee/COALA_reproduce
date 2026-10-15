---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: list[list[str]]
aliases: [el]
---
# entity_lists

← [[物件 總覽]]

**存什麼**：一個 batch 裡每句話各自的 [[entity_list]]。

| 項目 | 內容 |
|---|---|
| 型別 | `list[list[str]]`，外層長度 B，內層長度 10 |
| 由誰產生 | [[make_entity_lists()]]（[[train_stage1.py]]）、[[model.main()]] / [[overfit_test.main()]] 裡的 list comprehension |

## 去處
[[COALAModel.forward_stage1()]] 的第二個參數；迴圈中 `entity_lists[i]` 給第 i 句。

> [!note] 每個 step 重新抽
> 訓練時每步都用 [[rng]] 重抽干擾詞 → 同一句話每個 epoch 看到的干擾詞都不同（資料增強的效果）。[[validate()]] 則固定 `Random(0)`，讓每次驗證條件相同。
