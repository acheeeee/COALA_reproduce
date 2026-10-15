---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: build_manifest.py
---
# write_manifest()

← [[build_manifest.py]] · [[函式 總覽]]

**功能**：把 rows 寫成 TSV 檔。

```python
def write_manifest(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(FIELDS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in FIELDS) + "\n")
```

| | 物件 |
|---|---|
| 輸入 `rows` | [[row (manifest 一列)]] 的 list |
| 輸入 `out_path` | `Path`，例 `data/manifest/dev-clean.tsv` |
| 輸出 | 無（寫檔）→ [[manifest TSV]] |

用到：[[FIELDS]]（表頭與欄位順序）

- `mkdir(parents=True, exist_ok=True)`：資料夾不存在就建，已存在不報錯
- `str(r[k])`：數字也轉成字串才能 join

被 [[build_manifest.main()]] 呼叫。
