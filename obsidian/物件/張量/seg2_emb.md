---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (n2, 576)
dtype: float32
---
# seg2_emb

← [[物件 總覽]]

**存什麼**：[[seg2_ids]] 查表得到的向量。

| 項目 | 內容 |
|---|---|
| shape | `(n2, 576)` |
| 定義於 | [[build_stage1_sample()]]、[[build_gen_prompt()]] |

在 [[inputs_embeds]] 中排**第 3 段**（在 [[audio_tok]] 之後）。
