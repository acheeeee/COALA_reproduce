---
tags: [COALA, 函式, 資料層, biasing]
種類: 函式
檔案: biasing.py
---
# load_official_ref()

← [[biasing.py]] · [[函式 總覽]]

**功能**：讀官方 `ref/{split}.biasing_{n}.tsv`。

```python
def load_official_ref(split, n):
```

| | 型別 | 物件 |
|---|---|---|
| 輸入 `split` | `str` | 例 `"test-clean"` |
| 輸入 `n` | `int` | 100 / 500 / 1000 / 2000 |
| 輸出 | `dict` | [[ref (官方 ref dict)]] |

## 內部物件
`path` = [[OFFICIAL_REF_DIR]] `/ f"{split}.biasing_{n}.tsv"` → [[官方 ref TSV]]；每行 `split("\t")` 成 4 欄，後兩欄 `json.loads`。

```python
utt_id, text, pos_json, list_json = line.split("\t")
out[utt_id] = {"text": text,
               "positives": json.loads(pos_json),
               "biasing_list": json.loads(list_json)}
```

被 [[biasing.main()]] 呼叫。
