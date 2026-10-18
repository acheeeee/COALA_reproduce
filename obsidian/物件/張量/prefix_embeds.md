---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (P, 576)
dtype: float32
---
# prefix_embeds

← [[物件 總覽]]

**存什麼**：Stage 2 一句話的**共用前綴** —— 系統訊息 + 音訊 + 題目。121 條序列的這一段完全相同。

| 項目 | 內容 |
|---|---|
| shape | `(P, 576)`，P = 15 + L' + 13（實測假音訊 L' = 89 → **117**） |
| 由誰產生 | [[build_stage2_sample()]] |

## 長相
```
[ seg1_emb (15) | audio_tok (L') | inst_emb (13) ]
```
圖中物件：[[seg1_emb]]、[[audio_tok]]、[[STAGE2_INSTRUCTION]]

## 為什麼可以共用
SmolLM2 是 causal：前綴每個位置只看得到前面，看不到後面接哪個 entity → 121 條的前綴輸出完全一樣 → 只算一次（KV cache）。**雙向模型（BERT）就不行**：前綴會看到後面不同的 entity。

## 去處（8b，尚未實作）
`lm(inputs_embeds=prefix_embeds[None], use_cache=True)` → [[out (LM 輸出物件)]] 的 `past_key_values`，複製成 K 份給 [[suffix_embeds]] 接著用。
