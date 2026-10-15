---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (n1, 576)
dtype: float32
---
# seg1_emb

← [[物件 總覽]]

**存什麼**：[[seg1_ids]] 查 embedding 表得到的向量。

| 項目 | 內容 |
|---|---|
| shape | `(n1, 576)` = `(16, 576)` |
| 定義於 | [[build_stage1_sample()]]、[[build_gen_prompt()]] |

```python
seg1_emb = embed_layer(seg1_ids)
```

在 [[inputs_embeds]] 中排**第 1 段**：`[seg1_emb, audio_tok, seg2_emb, target_emb]`
