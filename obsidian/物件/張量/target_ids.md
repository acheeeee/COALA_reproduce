---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (m,)
dtype: int64
---
# target_ids

← [[物件 總覽]]

**存什麼**：正確逐字稿（加上結尾 `<|im_end|>`）的 token id —— LM 要學會產生的答案。

| 項目 | 內容 |
|---|---|
| shape | `(m,)`，例：dev-clean 第一句 m = 12 |
| dtype | `torch.long` |
| 定義於 | [[build_stage1_sample()]]（生成時沒有它） |

## 兩個去處
1. [[embed_layer]] 查表 → [[target_emb]]（teacher forcing 的輸入）
2. **直接放進 [[labels]] 的最後 m 格**（答案）

> [!danger] 長度要用 `.shape[0]`，不能用 `len(target_text)`
> 後者是**字元數**。曾經因此讓 [[attention_mask]] 長 209、[[inputs_embeds]] 長 160。
