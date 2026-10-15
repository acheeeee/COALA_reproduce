---
tags: [COALA, 物件, 張量, 音訊]
類別: 張量
型別: numpy.ndarray
shape: (n_samples,)
dtype: float64
aliases: [wave]
---
# wav

← [[物件 總覽]]

**存什麼**：一句話的原始波形，每個數字是一個取樣點的振幅。

| 項目 | 內容 |
|---|---|
| 型別 | `numpy.ndarray`（⚠️ 不是 torch tensor） |
| shape | `(n_samples,)`，`n_samples = 秒數 × 16000` |
| dtype | `float64`，值域約 −1 ~ 1 |
| 別名 | `wave`（[[check_shape.py]]） |
| 搭檔 | `sr`（取樣率，`int`，LibriSpeech 固定 **16000**） |

## 從哪來 → 到哪去
```
.flac 檔 ──sf.read()──→ wav, sr ──fe──→ input_features
```
圖中物件：[[fe]]、[[input_features]]
- 產生：[[LibriSpeechDataset.__getitem__()]]、[[check_shape.main()]]
- 唯一的去處：丟給 [[fe]]，之後就再也不會用到

例：10 秒的句子 → `(160000,)`。長度隨句子不同，這也是為什麼要轉成固定長度的 [[input_features]]。
