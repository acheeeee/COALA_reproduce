---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: dict
aliases: [r, rows]
---
# row (manifest 一列)

← [[物件 總覽]]

**存什麼**：[[manifest TSV]] 的一列 = 一句話的所有後設資料。同一個概念在三個檔案裡長得不太一樣。

## 版本 1：寫檔前（[[build_manifest()]]，`rows` 的元素）
```python
{"utt_id": "1272-128104-0000",
 "audio_path": "data/LIBRISPEECH/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac",
 "duration": 5.855,               # float，秒，round 到 3 位（示意值）
 "n_words": 17,                   # int
 "text": "mister quilter is ...", # 已小寫
 "rare_words": '["quilter", ...]'}  # ⚠️ JSON 字串，不是 list（示意）
```

## 版本 2：讀檔後（[[LibriSpeechDataset.__init__()]]，`self.rows` 的元素）
```python
{"utt_id": ..., "audio_path": ..., "duration": 5.855, "text": ...,
 "rare_words": ["quilter", ...]}  # 已 json.loads 成 list；沒有 n_words
```

## 版本 3：[[pack_two_utterances()]] 的 `rows` 元素
`(utt_id, audio_path, duration)` 三元組（tuple）

## 欄位順序
由 [[FIELDS]] 決定。
