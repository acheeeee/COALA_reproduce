---
tags: [COALA, 物件, 張量, 音訊]
類別: 張量
型別: torch.Tensor
shape: (80, 3000) / (B, 80, 3000)
dtype: float32
aliases: [feats, mel]
---
# input_features

← [[物件 總覽]]

**存什麼**：Whisper 格式的 log-Mel 頻譜圖 —— 每 10 ms 一個 frame，每個 frame 是 80 個頻帶的能量。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor` |
| shape | 單筆 `(80, 3000)`；batch `(B, 80, 3000)` |
| dtype | `float32` |
| 別名 | `feats`（[[COALAModel.forward_stage1()]]、[[transcribe_one()]]）、`mel`（[[check_shape.py]]） |

## 維度意義

| 維度 | 大小 | 意義 |
|---|---|---|
| B | batch size | 幾句話 |
| 80 | n_mels | Mel 頻帶數（large-v2 是 80；v3 才是 128） |
| 3000 | frames | **固定 30 秒** × 100 frame/秒。短的句子後面補 0，長的截掉 |

## 從哪來 → 到哪去
```
wav ──fe──→ (1, 80, 3000) ──[0]──→ (80, 3000)          ← __getitem__
                                    ──torch.stack──→ (B, 80, 3000) ← collate_fn
                                    ──encoder──→ x (encoder 輸出) (B, 1500, 1280)
```
圖中物件：[[wav]]、[[fe]]、[[encoder]]、[[x (encoder 輸出)]]
- 產生：[[LibriSpeechDataset.__getitem__()]] → [[collate_fn()]] 疊成 batch
- 使用：[[COALAModel.encode_audio()]]；[[transcribe_one()]] 用 `.unsqueeze(0)` 補回 batch 維 → `(1, 80, 3000)`

> [!tip] 為什麼可以直接 `torch.stack`
> 因為每句都被 pad 成一樣的 3000 frames，不像 [[ctc_ids]] 要另外 pad。代價是短句也要算滿 30 秒的 encoder。
