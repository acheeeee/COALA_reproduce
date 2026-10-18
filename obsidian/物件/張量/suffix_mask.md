---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (K, S)
dtype: int64
aliases: [entity_mask]
---
# suffix_mask

← [[物件 總覽]]

**存什麼**：K 條尾巴裡哪些位置是真的 entity token（1）、哪些是 pad（0）。

| 項目 | 內容 |
|---|---|
| shape | `(K, S)` |
| dtype | `torch.long` |
| 由誰產生 | [[build_stage2_sample()]] |

## 兩個用途（8b）
1. **attention mask**：前面接上前綴的 P 個 1 → `cat([ones(K, P), suffix_mask], dim=1)`，告訴 LM 哪些尾巴位置是 pad
2. **projector 的 mask**：因為尾巴**只放 entity**，這個遮罩剛好就是 entity 的位置 → `.float()` 後直接當 [[mask (projector)]]，做 Eq. 1 的長度正規化

`suffix_mask[i].sum()` = 第 i 個候選的 token 數 Lᵢ。

> 目前設計下 suffix_mask = entity_mask。如果以後在 entity 後面再接其他 token（例如 `<|im_end|>`），兩者就要分開。
