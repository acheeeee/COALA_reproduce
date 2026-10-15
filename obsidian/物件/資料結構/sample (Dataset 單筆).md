---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: dict
aliases: [s]
---
# sample (Dataset 單筆)

← [[物件 總覽]]

**存什麼**：`ds[i]` 拿到的一句話，已經轉成模型能吃的格式。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[LibriSpeechDataset.__getitem__()]] |
| 別名 | `s`（[[dataset.main()]]、[[evaluate.main()]]） |

```python
{"utt_id":         "1272-128104-0000",       # str
 "input_features": Tensor (80, 3000),        # float32
 "text":           "mister quilter is ...",  # str
 "ctc_ids":        Tensor (n_chars,),        # long
 "rare_words":     ["quilter", ...]}         # list[str]（示意）
```
→ [[utt_id]]、[[input_features]]、[[text]]、[[ctc_ids]]、[[positives (E⁺)]]

## 去處
- [[collate_fn()]] 把 B 個疊成 [[batch]]
- [[evaluate.main()]] 不經過 DataLoader，直接逐句取用
