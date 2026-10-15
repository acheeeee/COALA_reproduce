---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[str]
aliases: [negatives, E-]
---
# negatives (E⁻)

← [[物件 總覽]]

**存什麼**：隨機抽來、**不在這句話裡**的 rare word = 論文的**負樣本 E⁻**（干擾詞）。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]`，長度 = `n_distractors` |
| 定義於 | [[make_biasing_list()]] 的區域變數 |

```python
sampled   = rng.sample(rare_pool, n_distractors + len(pos))   # 多抽 len(pos) 個
negatives = [w for w in sampled if w not in pos_set][:n_distractors]
```
多抽的那幾個是保險：最壞情況全部正樣本都被抽到，濾掉後仍然夠，不需要 while 迴圈重抽。

和 [[positives (E⁺)]] 合併排序 → [[biasing_list]]。

> 完整的 E⁻（含條件性的 [[UNBIASED]]）是用 [[neg_idx]] 表示的索引，不是這個 list。
