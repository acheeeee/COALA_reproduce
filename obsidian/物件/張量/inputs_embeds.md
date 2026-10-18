---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (tot, 576) / (B, max_tot, 576) / (1, n1+L'+n2, 576)
dtype: float32
aliases: [embeds]
---
# inputs_embeds

← [[物件 總覽]]

**存什麼**：真正餵進 LM 的整條輸入序列 —— 文字向量和音訊向量拼在一起。

| 版本 | shape | 由誰產生 |
|---|---|---|
| 單句訓練 | `(tot, 576)`，tot = n1 + L' + n2 + m | [[build_stage1_sample()]] |
| batch 訓練 | `(B, max_tot, 576)`，右側補 0 | [[pad_batch()]]（區域變數名 `embeds`） |
| 生成 | `(1, n1 + L' + n2, 576)` | [[build_gen_prompt()]] |

## 長相
```
[ seg1_emb | audio_tok | seg2_emb | target_emb ]
   15          L'         n2          m
例：15 +      89     +   44   +     12  = 160 / 2048（context window 只用 8%）
```
→ [[seg1_emb]]、[[audio_tok]]、[[seg2_emb]]、[[target_emb]]

## 為什麼用 inputs_embeds 而不是 input_ids
[[audio_tok]] 是 576 維向量，**沒有 token id**。只能自己查表把文字也變成向量，再 `torch.cat` 起來，用 `lm(inputs_embeds=...)` 餵進去。LM 的 Transformer 層本來就只看向量，不知道「token」這回事。

## 去處
- `self.lm(**padded)`（[[COALAModel.forward_stage1()]]）
- `model.lm.generate(inputs_embeds=...)`（[[transcribe_one()]]）

三個一組：[[inputs_embeds]]、[[labels]]、[[attention_mask]] 的長度必須完全相同。
