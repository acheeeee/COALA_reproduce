---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: dict
---
# sample (stage1 序列)

← [[物件 總覽]]

**存什麼**：一句話組好、可以直接餵 LM 的三件組（還沒 pad）。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[build_stage1_sample()]] |

| 鍵 | shape | 物件 |
|---|---|---|
| `inputs_embeds` | `(tot, 576)` | [[inputs_embeds]] |
| `labels` | `(tot,)` | [[labels]] |
| `attention_mask` | `(tot,)` | [[attention_mask]] |

tot = n1 + L' + n2 + m，**每句不同**。

> 和 [[sample (Dataset 單筆)]] 不同：那個是原始資料，這個是組好的 LM 輸入。

收集成 [[samples]] → [[pad_batch()]] → [[padded]]
