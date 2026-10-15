---
tags: [COALA, 函式, 訓練]
種類: 函式
檔案: train_stage1.py
---
# make_entity_lists()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：對 batch 裡每一句各產生 10 個 entity。

```python
def make_entity_lists(rare_words_batch, rare_pool, rng):
    return [make_stage1_list(rw, rare_pool, rng) for rw in rare_words_batch]
```

| 參數 | 物件 |
|---|---|
| `rare_words_batch` | `batch["rare_words"]`：每句的 [[positives (E⁺)]] |
| `rare_pool` | [[rare_pool]] |
| `rng` | [[rng]] |
| **輸出** | [[entity_lists]] |

呼叫：[[make_stage1_list()]]

被 [[train_stage1.main()]]（每步）、[[validate()]] 呼叫。
