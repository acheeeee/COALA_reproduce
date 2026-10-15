---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# OFFICIAL_REF_DIR

← [[物件 總覽]]

**存什麼**：官方測試用 biasing list 的資料夾。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | `/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref` |
| 定義於 | [[biasing.py]] |

## 用在哪裡
- [[load_official_ref()]]：讀 `OFFICIAL_REF_DIR / f"{split}.biasing_{n}.tsv"` → [[官方 ref TSV]]

只有 N = 100 / 500 / 1000 / 2000，**沒有論文用的 N=5000**。
