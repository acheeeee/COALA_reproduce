---
tags: [COALA, 物件, 資料結構, Stage2]
類別: 資料結構
型別: dict
aliases: [s2]
---
# sample (stage2 序列)

← [[物件 總覽]]

**存什麼**：一句話的 Stage 2 輸入，前綴和尾巴分開放。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[build_stage2_sample()]] |
| 別名 | `s2`（[[prompt.main()]]） |

| 鍵 | shape | 物件 | 給 8b 做什麼 |
|---|---|---|---|
| `prefix_embeds` | `(P, 576)` | [[prefix_embeds]] | 前綴只算一次 |
| `suffix_embeds` | `(K, S, 576)` | [[suffix_embeds]] | K 條尾巴一起算 |
| `suffix_mask` | `(K, S)` | [[suffix_mask]] | attention mask + projector mask |
| `entity_ids` | `(K, S)` | [[entity_ids]] | log loss 的答案 |

> 和 [[sample (stage1 序列)]] 的差別：Stage 1 是拼好的一整條 + labels；Stage 2 刻意**不拼**，讓前綴可以共用。
