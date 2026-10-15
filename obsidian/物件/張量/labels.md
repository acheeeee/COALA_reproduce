---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (tot,) / (B, max_tot)
dtype: int64
---
# labels

← [[物件 總覽]]

**存什麼**：每個位置的「正確答案 token id」，不算 loss 的位置填 [[IGNORE_INDEX]]（-100）。

| 版本 | shape | 由誰產生 |
|---|---|---|
| 單句 | `(tot,)` | [[build_stage1_sample()]] |
| batch | `(B, max_tot)`，pad 處 -100 | [[pad_batch()]] |

## 長相
```
inputs_embeds: [ seg1 | audio | seg2 | target ]
labels:        [ -100 | -100  | -100 | target_ids ]
                 ←── n1 + L' + n2 ──→   ← m →
```
例：160 個位置中只有 12 個算 loss。

## 不用自己 shift
HF 的 LM 收到 labels 會在內部把 logits 和 labels 錯開一位（位置 t 的輸出去預測位置 t+1 的 label），所以 labels 和 [[inputs_embeds]] 直接對齊即可。

## 去處
- `self.lm(**padded)` → [[ce_loss]]
