---
tags: [COALA, 函式, 資料層, biasing]
種類: 函式
檔案: biasing.py
---
# make_stage1_list()

← [[biasing.py]] · [[函式 總覽]]

**功能**：產生 Stage 1 prompt 用的 10 個 entity —— 最多 5 個正樣本，其餘補干擾詞。

```python
def make_stage1_list(positives, rare_pool, rng, max_pos=5, n_total=10):
```

| | 名稱 | 物件 |
|---|---|---|
| 輸入 | `positives` | [[positives (E⁺)]] |
| 輸入 | `rare_pool` | [[rare_pool]] |
| 輸入 | `rng` | [[rng]] |
| 輸入 | `max_pos=5`、`n_total=10` | 論文 3.4 |
| 輸出 | `list[str]`，長度 10 | [[entity_list]] |

```python
pos = sorted(set(positives))
if len(pos) > max_pos:
    pos = sorted(rng.sample(pos, max_pos))    # 超過 5 個就隨機挑 5 個
return make_biasing_list(pos, rare_pool, n_total - len(pos), rng)
```

## 呼叫了誰
[[make_biasing_list()]]

## 被誰呼叫
[[make_entity_lists()]]、[[evaluate.main()]]（`--mode stage1`）、[[overfit_test.main()]]、[[model.main()]]

> [!important] 為什麼刻意不給全部正樣本
> 論文 3.4：*sampling up to 5 positive entities within a 10-entity set*。如果 prompt 永遠包含所有答案，模型會變成「照抄清單」而不是聽音訊。
