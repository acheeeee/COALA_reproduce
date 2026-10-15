---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (tot,) / (B, max_tot) / (1, total)
dtype: int64
aliases: [mask]
---
# attention_mask

← [[物件 總覽]]

**存什麼**：哪些位置是真的內容（1）、哪些是 padding（0）。

| 版本 | shape | 內容 | 由誰產生 |
|---|---|---|---|
| 單句 | `(tot,)` | 全 1 | [[build_stage1_sample()]] |
| batch | `(B, max_tot)` | 真內容 1、pad 0 | [[pad_batch()]]（區域變數名 `mask`） |
| 生成 | `(1, total)` | 全 1 | [[build_gen_prompt()]] |

## 例
```
pad_side="right"：[1 1 1 1 1 1 0 0 0]   ← 訓練用
pad_side="left" ：[0 0 0 1 1 1 1 1 1]   ← 生成用（新 token 要接在最右邊）
```

> 和 [[mask (projector)]] 不同：那個標的是「entity 的哪些 token 有效」，給長度正規化用。
