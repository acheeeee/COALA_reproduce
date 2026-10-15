---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# LIBRISPEECH_ROOT

← [[物件 總覽]]

**存什麼**：LibriSpeech 語料根目錄（相對路徑，透過 symlink 指到 `/mnt/disk2/LIBRISPEECH`）。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | `data/LIBRISPEECH/LibriSpeech` |
| 定義於 | [[build_manifest.py]] |

## 用在哪裡
- [[build_manifest()]]：`LIBRISPEECH_ROOT / split` → 走訪該 split 的所有 [[trans.txt]]

相對路徑 → 必須從 repo 根目錄執行。
