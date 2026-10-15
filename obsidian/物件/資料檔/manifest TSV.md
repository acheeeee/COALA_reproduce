---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: data/manifest/{split}.tsv
---
# manifest TSV

← [[物件 總覽]]

**存什麼**：每個 split 一張表，一列一句話 —— 之後所有程式都只讀它，不再走訪 LibriSpeech 目錄。

| 項目 | 內容 |
|---|---|
| 格式 | tab 分隔，第一列是表頭（[[FIELDS]]） |
| 一列 | [[row (manifest 一列)]] |
| 寫入 | [[build_manifest.main()]] → [[write_manifest()]] |
| 讀取 | [[LibriSpeechDataset.__init__()]]、[[pack_two_utterances()]] |

```
utt_id	audio_path	duration	n_words	text	rare_words
1272-128104-0000	data/.../1272-128104-0000.flac	…	17	mister quilter is …	["quilter", …]
```

## 已建立的 split（實測）

| split | 句數 | 時數 | 無 rare word | 多目標（≥2） |
|---|---|---|---|---|
| test-clean | 2620 | 5.40 h | 24.43% | 49.20% |
| train-clean-100 | 28539 | 100.59 h | 7.42% | 78.43% |
| 全部訓練集 | 281,241 | 961.06 h | | |

[[MANIFEST_DIR]] 是輸出資料夾。
