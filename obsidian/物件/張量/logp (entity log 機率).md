---
tags: [COALA, 物件, 張量, Stage2, loss]
類別: 張量
型別: torch.Tensor
shape: (K,)
dtype: float32
aliases: [logp]
---
# logp (entity log 機率)

← [[物件 總覽]]

**存什麼**：每個候選 entity 被 LM「整串生成出來」的 log 機率 log P(eᵢ | X) —— entity 所有 token 的 log 機率加總。

| 項目 | 內容 |
|---|---|
| shape | `(K,)` |
| 值域 | ≤ 0 |
| 由誰產生 | 8b 的 `forward_stage2`（**尚未實作**）：lm_head 的 logits + [[entity_ids]] |
| 用在 | [[bias_loss()]] 的 log loss |

## 怎麼算（8b 要注意的地方）
LM 在位置 t 的輸出預測的是位置 t+1。所以 entity **第一個** token 的機率，來自**前綴最後一個位置**（題目的 `:`）的輸出，不在尾巴裡。

## 測試用的假值
[[losses.main()]]：`[−5.0, −2.0, −3.0, −8.0, −9.0]`

> 和 [[s (entity 分數)]] 走不同的路：s 經過 projector（判別空間），logp 經過 lm_head（49152 詞彙空間）。
