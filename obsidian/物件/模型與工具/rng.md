---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: random.Random
---
# rng

← [[物件 總覽]]

**是什麼**：獨立的亂數產生器，專門給「抽干擾詞」用。

| 項目 | 內容 |
|---|---|
| 型別 | `random.Random(seed)` |
| 用在 | [[make_biasing_list()]]、[[make_stage1_list()]] 的 `rng.sample(...)` |

## 各處的 seed

| 地方 | seed | 為什麼 |
|---|---|---|
| [[train_stage1.main()]] | `args.seed`（42） | 訓練可重現 |
| [[validate()]] | **固定 0**，每次驗證重建 | 每次驗證用同一組干擾詞，val loss 才能互相比較 |
| [[evaluate.main()]] | `args.seed` | |
| [[biasing.main()]] | 42、42、43 | 測試「同 seed 結果相同、不同 seed 結果不同」 |

## 為什麼不用全域的 `random`
`random.Random(42)` 是獨立物件，不會被其他程式碼（例如 DataLoader 的 shuffle）用掉亂數而改變序列。當參數傳進函式，就能精確控制「這次用哪個 seed」。

> 論文**沒有公開 seed** → 自己的 seed 必須固定並記錄。
