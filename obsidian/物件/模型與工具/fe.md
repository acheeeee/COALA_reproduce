---
tags: [COALA, 物件, 模型與工具, 音訊]
類別: 模型與工具
型別: transformers.WhisperFeatureExtractor
aliases: [feature_extractor]
---
# fe

← [[物件 總覽]]

**是什麼**：Whisper 的前處理器，把波形變成 log-Mel 頻譜圖。沒有可訓練參數，是固定的數學轉換。

| 項目 | 內容 |
|---|---|
| 型別 | `WhisperFeatureExtractor` |
| 建立 | `WhisperFeatureExtractor.from_pretrained(WHISPER_ID)` |
| 別名 | `self.fe`、參數名 `feature_extractor`（[[LibriSpeechDataset.__init__()]]） |

## 做了什麼
```
wav (n_samples,) @16kHz
  → pad / 截斷成 30 秒（480,000 點）
  → STFT（每 10 ms 一個 frame）→ 80 個 Mel 頻帶 → log → 正規化
  → BatchFeature，.input_features 是 (1, 80, 3000)
```

```python
out = self.fe(wav, sampling_rate=sr, return_tensors="pt")
input_features = out.input_features[0]       # 去掉 batch 維
```
`return_tensors="pt"` = 回傳 PyTorch tensor（不是 numpy）。

## 誰在用
[[LibriSpeechDataset.__getitem__()]]、[[check_shape.main()]]；在各個 main 裡建立後傳給 Dataset。

輸入 [[wav]] → 輸出 [[input_features]]
