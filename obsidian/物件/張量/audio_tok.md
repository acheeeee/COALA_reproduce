---
tags: [COALA, 物件, 張量, 音訊]
類別: 張量
型別: torch.Tensor
shape: (L', 576)
dtype: float32
aliases: [audio_tokens]
---
# audio_tok

← [[物件 總覽]]

**存什麼**：CTC 挑選後留下的音訊向量，論文圖 2(a) 的 **Audio Tokens**。每個大約對應一個預測出的字元。

| 項目 | 內容 |
|---|---|
| shape | `(L', 576)`，**L' 每句不同**（≈ CTC 預測出的字元數） |
| dtype | `float32`，帶梯度 |
| 別名 | `audio_tokens`（[[build_stage1_sample()]]、[[build_gen_prompt()]] 的參數名） |
| 注意 | **沒有 batch 維**，一次只處理一句 |

## 從哪來 → 到哪去
```
h (adapter 輸出)[i] (750, 576) ──ctc_compress() 用 keep 挑列──→ audio_tok (L', 576)
     ──build_stage1_sample()──→ 夾在 seg1_emb 和 seg2_emb 中間 → inputs_embeds
```
圖中物件：[[h (adapter 輸出)]]、[[ctc_compress()]]、[[keep]]、[[build_stage1_sample()]]、[[seg1_emb]]、[[seg2_emb]]、[[inputs_embeds]]

## L' 的意義
理想上 L' ≈ 這句的字元數（例：dev-clean 第一句 89）。
- stack=4 實測 L' 只有字元數的 **0.33**（CTC 沒學好 → 84% 資訊被丟）
- stack=2 的比值 **尚未量測**（STATUS 待辦 #3）

> [!note] 為什麼要逐句處理
> L' 隨內容變化，batch 內每句長度不同 → 無法一起算，只能 for 迴圈逐句，最後用 [[pad_batch()]] 補齊。
