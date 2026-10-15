---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (m, 576)
dtype: float32
---
# target_emb

← [[物件 總覽]]

**存什麼**：[[target_ids]] 查表得到的向量 —— 訓練時把「正確答案」也當輸入餵進去（teacher forcing）。

| 項目 | 內容 |
|---|---|
| shape | `(m, 576)` |
| 定義於 | [[build_stage1_sample()]] |

在 [[inputs_embeds]] 中排**第 4 段**（最後）。生成時（[[build_gen_prompt()]]）沒有這一段。
