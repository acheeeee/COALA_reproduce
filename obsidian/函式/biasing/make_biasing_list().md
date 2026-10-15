---
tags: [COALA, 函式, 資料層, biasing]
種類: 函式
檔案: biasing.py
---
# make_biasing_list()

← [[biasing.py]] · [[函式 總覽]]

**功能**：正樣本 + 隨機抽的干擾詞 → 字母排序的 biasing list（不含 `<unbiased>`）。

```python
def make_biasing_list(positives, rare_pool, n_distractors, rng):
```

## 輸入 → 輸出

| | 名稱 | 型別 | 物件 |
|---|---|---|---|
| 輸入 | `positives` | `list[str]` | [[positives (E⁺)]] |
| 輸入 | `rare_pool` | `list[str]` | [[rare_pool]] |
| 輸入 | `n_distractors` | `int` | 要幾個干擾詞（評估時 N 個） |
| 輸入 | `rng` | `random.Random` | [[rng]] |
| 輸出 | | `list[str]` | [[biasing_list]]，長度 `len(positives) + n_distractors` |

## 內部物件

| 名稱 | 角色 |
|---|---|
| `pos` | `sorted(set(positives))`：去重 + 排序 |
| `pos_set` | `set(pos)`：O(1) 查詢用 |
| `sampled` | 抽 `n_distractors + len(pos)` 個（**多抽保險**） |
| `negatives` | 濾掉正樣本後取前 `n_distractors` 個 → [[negatives (E⁻)]] |

## 原始碼
```python
pos = sorted(set(positives))
pos_set = set(pos)
sampled = rng.sample(rare_pool, n_distractors + len(pos))
negatives = [w for w in sampled if w not in pos_set][:n_distractors]
return sorted(pos + negatives)
```

## 被誰呼叫
[[make_stage1_list()]]、[[biasing.main()]]；之後 Stage 2（M=120）和評估（N=500/1000/5000）也會用它。

> [!tip] 為什麼多抽 len(pos) 個
> 最壞情況：抽到的裡面剛好包含所有正樣本。多抽 `len(pos)` 個，濾掉後至少還剩 `n_distractors` 個 → 一次保證抽足，不用 while 迴圈。
