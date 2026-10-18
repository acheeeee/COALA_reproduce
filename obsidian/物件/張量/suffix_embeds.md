---
tags: [COALA, 物件, 張量, Stage2]
類別: 張量
型別: torch.Tensor
shape: (K, S, 576)
dtype: float32
---
# suffix_embeds

← [[物件 總覽]]

**存什麼**：K 條尾巴，每條只有一個候選 entity 的 token 向量，右側補 0。

| 項目 | 內容 |
|---|---|
| shape | `(K, S, 576)`，K = 候選數（訓練 121），S = 最長候選的 token 數 |
| pad 值 | 0 |
| 由誰產生 | [[build_stage2_sample()]] |

## 例（[[prompt.main()]]，K = 4，S = 4）
```
第 0 列  " <unbiased>"   [tok tok tok tok]
第 1 列  " watson"       [tok tok  0   0 ]
第 2 列  " solved"       [tok  0   0   0 ]
第 3 列  " craswellers"  [tok tok tok tok]
```

## 去處（8b）
接在 [[prefix_embeds]] 的 KV cache 後面送進 LM → 最後一層 hidden state `(K, S, 576)` = [[h (projector 輸入)]]

搭配 [[suffix_mask]]（哪些是真 token）、[[entity_ids]]（token 編號）。
