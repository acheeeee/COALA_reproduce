---
tags: [COALA, 函式, 資料層]
種類: 方法
檔案: dataset.py
---
# LibriSpeechDataset.__getitem__()

← [[LibriSpeechDataset]] · [[dataset.py]] · [[函式 總覽]]

**功能**：取第 i 句：讀音檔 → log-Mel，逐字稿 → CTC id。

```python
def __getitem__(self, i):
```

## 輸入 → 輸出

| | 物件 |
|---|---|
| 輸入 `i` | `int`，第幾句 |
| 輸出 | [[sample (Dataset 單筆)]] |

## 資料流
```
self.rows[i] (r)
 ├─ r["audio_path"] ──sf.read──→ wav, sr ──fe──→ (1,80,3000) ──[0]──→ input_features (80,3000)
 ├─ r["text"] ──text_to_ctc_ids()──→ id_list ──torch.tensor(long)──→ ctc_ids (n_chars,)
 ├─ r["utt_id"]     → 原樣
 ├─ r["text"]       → 原樣
 └─ r["rare_words"] → 原樣（已是 list）
```
圖中物件：[[wav]]、[[fe]]、[[input_features]]、[[text_to_ctc_ids()]]、[[ctc_ids]]

## 內部物件
`r`（[[row (manifest 一列)]]）、[[wav]]、`sr`、`out`（fe 回傳的 BatchFeature）、[[input_features]]、`id_list`、[[ctc_ids]]

## 呼叫了誰
`sf.read`、[[fe]]、[[text_to_ctc_ids()]]

## 被誰呼叫
[[DataLoader (dl)]]（`ds[i]`）、[[evaluate.main()]]、[[dataset.main()]]

> 這是整個訓練最耗 CPU 的地方（解碼 flac + 算頻譜），所以 DataLoader 開 8 個 workers 平行做。
