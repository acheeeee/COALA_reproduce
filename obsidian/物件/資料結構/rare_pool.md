---
tags: [COALA, 物件, 資料結構, 詞表]
類別: 資料結構
型別: list[str]
---
# rare_pool

← [[物件 總覽]]

**存什麼**：所有 rare word 的詞池，用來隨機抽干擾詞 [[negatives (E⁻)]]。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]`，**排序過**，209,291 個（= 論文說的 209.2K） |
| 來源 | [[all_rare_words.txt]]，經 [[load_rare_pool()]] 讀入 |

## 為什麼是排序過的 list 而不是 set
`rng.sample()` 需要有順序的容器。對 set 抽樣，結果會隨 Python 內部 hash 順序變動 → 同一個 seed 每次執行結果不同 → **破壞可重現性**。

## 誰在用
- [[make_biasing_list()]]、[[make_stage1_list()]]
- [[make_entity_lists()]]、[[validate()]]、[[train_stage1.main()]]、[[evaluate.main()]]、[[overfit_test.main()]]、[[model.main()]]
