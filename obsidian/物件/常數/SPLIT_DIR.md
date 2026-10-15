---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# SPLIT_DIR

← [[物件 總覽]]

**存什麼**：[[figure1.py]] 要統計的 split 目錄。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | `data/LIBRISPEECH/LibriSpeech/dev-clean` |
| 定義於 | [[figure1.py]] |

## 用在哪裡
- [[figure1.main()]]：`SPLIT_DIR.rglob("*.trans.txt")` 找出全部 [[trans.txt]]
