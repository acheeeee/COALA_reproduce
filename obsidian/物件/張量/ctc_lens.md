---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (B,)
dtype: int64
aliases: [target_lengths]
---
# ctc_lens

← [[物件 總覽]]

**存什麼**：batch 裡每句話真實的字元數（= pad 前 [[ctc_ids]] 的長度）。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor` |
| shape | `(B,)` |
| dtype | `torch.long` |
| 別名 | `target_lengths`（[[compute_ctc_loss()]] 的參數名） |
| 實測例 | dev-clean 前 4 句 `[89, 63, 172, 133]` |

## 從哪來 → 到哪去
- 產生：[[collate_fn()]]
- 使用：[[compute_ctc_loss()]]；`max()` 決定 [[ctc_ids]] 要 pad 到多長

> [!important] CTC 的硬約束
> 每句都要滿足 **位置數 ≥ ctc_lens[i]**（位置數 = [[h (adapter 輸出)]] 的長度）。stack=4 時位置只有 375，83.2% 的句子違反 → loss = inf → 被 `zero_infinity` 歸零。
