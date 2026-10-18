---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (n2,)
dtype: int64
---
# seg2_ids

← [[物件 總覽]]

**存什麼**：[[seg2]] 文字（指令 + entity 清單 + assistant turn 開頭）的 token id。

| 項目 | 內容 |
|---|---|
| shape | `(n2,)`，長度隨 [[entity_list]] 而變（10 個 entity 時實測 n1+n2 = 59 → n2 = 44） |
| dtype | `torch.long` |
| 定義於 | [[build_stage1_sample()]]、[[build_gen_prompt()]] |

→ [[embed_layer]] 查表 → [[seg2_emb]]
