---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[str]
aliases: [lst_a, s1]
---
# biasing_list

← [[物件 總覽]]

**存什麼**：一句話的候選 entity 清單 = E⁺ ∪ E⁻，**不含** `<unbiased>`。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]`，字母排序 |
| 長度 | `len(positives) + n_distractors` |
| 由誰產生 | [[make_biasing_list()]]；[[make_stage1_list()]] 是長度固定 10 的特例 |

## 為什麼要字母排序
如果正樣本永遠排在前面，模型可以靠「位置」作弊，不用真的聽音訊。

## 去處
- [[add_unbiased()]] → [[full_list (E)]]
- Stage 1 的 10 個 entity 版本就是 [[entity_list]]

## 各階段的長度

| 用途 | 長度 |
|---|---|
| Stage 1 prompt | 10（最多 5 個正樣本） |
| Stage 2 訓練 | M = 120（+ `<unbiased>` = 121） |
| 評估 | N = 500 / 1000 / 5000 |
