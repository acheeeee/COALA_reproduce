---
tags: [COALA, 物件, 常數]
類別: 常數
型別: list[str]
---
# FIELDS

← [[物件 總覽]]

**存什麼**：[[manifest TSV]] 的欄位順序（= 表頭）。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]`，長度 6 |
| 值 | `["utt_id", "audio_path", "duration", "n_words", "text", "rare_words"]` |
| 定義於 | [[build_manifest.py]] |

## 用在哪裡
- [[write_manifest()]]：寫表頭，並照這個順序把每個 [[row (manifest 一列)]] 的值接成一行
- 讀檔端（[[LibriSpeechDataset.__init__()]]）**不依賴順序**，而是讀表頭建 `{欄名: 欄位索引}` → 欄位順序改了也不會壞
