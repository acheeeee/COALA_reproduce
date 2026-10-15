---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: str
---
# utt_id

← [[物件 總覽]]

**存什麼**：一句話的唯一編號，整個 pipeline 用它對齊「音檔、逐字稿、預測、官方答案」。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 格式 | `{speaker}-{chapter}-{句號}`，例 `"1272-128104-0000"` |
| 對應音檔 | `.../{speaker}/{chapter}/{utt_id}.flac` |

## 流經的地方
[[trans.txt]] 每行開頭 → [[items (trans 解析結果)]] → [[row (manifest 一列)]] → [[sample (Dataset 單筆)]] → [[batch]]`["utt_id"]` → [[hyp TSV]] 的第一欄 → 官方 `score.py` 用它和 [[官方 ref TSV]] 對齊
