---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (K, S)
dtype: int64
---
# entity_ids

← [[物件 總覽]]

**存什麼**：K 個候選 entity 的 token 編號，pad 位置填 [[IGNORE_INDEX]]（−100）。

| 項目 | 內容 |
|---|---|
| shape | `(K, S)` |
| dtype | `torch.long` |
| pad 值 | **−100** |
| 由誰產生 | [[build_stage2_sample()]] |

## 例
```
" watson"  → [267, 19201, -100, -100]
" solved"  → [14168, -100, -100, -100]
```

## 用途：算 log loss 的答案（8b）
```python
F.cross_entropy(logits, entity_ids, ignore_index=IGNORE_INDEX)
```
−100 的位置自動跳過，和 Stage 1 的 [[labels]] 同一個機制。

> [!danger] pad 不能填 0
> 0 是 SmolLM2 詞彙裡真的 token（`<|endoftext|>`）。填 0 等於教模型「在 pad 位置預測 `<|endoftext|>`」，錯了也不會報錯。

→ 算出 [[logp (entity log 機率)]]
