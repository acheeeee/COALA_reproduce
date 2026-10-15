---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# MANIFEST_DIR

← [[物件 總覽]]

**存什麼**：manifest TSV 的輸出資料夾。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | `data/manifest` |
| 定義於 | [[build_manifest.py]] |

## 用在哪裡
- [[build_manifest.main()]]：`MANIFEST_DIR / f"{split}.tsv"` → 寫出 [[manifest TSV]]
