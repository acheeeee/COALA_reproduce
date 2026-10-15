---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: dict
aliases: [ref]
---
# ref (官方 ref dict)

← [[物件 總覽]]

**存什麼**：讀進記憶體的官方測試 biasing list。

| 項目 | 內容 |
|---|---|
| 型別 | `dict[str, dict]` |
| 由誰產生 | [[load_official_ref()]]，讀 [[官方 ref TSV]] |

```python
{"1089-134686-0000": {
    "text": "he hoped there would be stew for dinner ...",
    "positives": [...],                             # 官方 E⁺
    "biasing_list": [...]},                         # 官方完整清單（不含 <unbiased>）
 ...}
```

## 誰在用
[[biasing.main()]]：拿官方的 `positives` 和自己的 [[find_rare_words()]] 比對 → 2620/2620 一致。

之後評估時應直接用官方清單，而不是自己抽，數字才能和論文比。
