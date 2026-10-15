---
tags: [COALA, 函式, 資料層, biasing]
種類: 函式
檔案: biasing.py
---
# split_pos_neg()

← [[biasing.py]] · [[函式 總覽]]

**功能**：依論文 2.1，切出完整清單 E 中哪些索引是 E⁺、哪些是 E⁻。**三個 loss 都依賴它。**

```python
def split_pos_neg(full_list, positives):
```

## 輸入 → 輸出

| | 名稱 | 物件 |
|---|---|---|
| 輸入 | `full_list` | [[full_list (E)]]（含 `<unbiased>`） |
| 輸入 | `positives` | [[positives (E⁺)]]（可能是空的） |
| 輸出 | `(pos_idx, neg_idx)` | [[pos_idx]]、[[neg_idx]] |

## 原始碼
```python
pos_set = set(positives)
if pos_set:                                              # 有 rare word
    pos_idx = [i for i, e in enumerate(full_list) if e in pos_set]
else:                                                    # 沒有 rare word
    pos_idx = [full_list.index(UNBIASED)]
pos_idx_set = set(pos_idx)
neg_idx = [i for i in range(len(full_list)) if i not in pos_idx_set]
return pos_idx, neg_idx
```

## 兩種情況

| 情況 | E⁺ | `<unbiased>` |
|---|---|---|
| 有 rare word | 那些 rare word | 屬於 E⁻ |
| 沒有 rare word | `{<unbiased>}` | 唯一正樣本 |

用到：[[UNBIASED]]

被 [[biasing.main()]] 呼叫；Stage 2 與 `losses.py` 會用。
