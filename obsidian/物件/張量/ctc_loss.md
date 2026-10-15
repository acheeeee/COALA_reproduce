---
tags: [COALA, 物件, 張量, loss]
類別: 張量
型別: torch.Tensor
shape: () 純量
dtype: float32
---
# ctc_loss

← [[物件 總覽]]

**存什麼**：CTC loss —— 「音訊位置序列」能多好地對齊到正確字元序列。

| 項目 | 內容 |
|---|---|
| shape | `()`（0 維純量，帶梯度） |
| 定義於 | [[compute_ctc_loss()]] 回傳；[[COALAModel.forward_stage1()]] 接收 |
| reduction | `"mean"`：每句 loss 先除以該句目標長度，再對 batch 平均 |
| `zero_infinity=True` | 無解（位置數 < 字元數）的句子 loss = inf → 歸零跳過 |

## 量級參考

| 狀態 | 值 |
|---|---|
| 隨機初始化（batch 89/63/172/133） | 約 10.18 |
| stack=4 收斂 | 2.70（卡住，欠擬合） |
| **stack=2 收斂** | **0.775** |

## 去處
- 乘上 [[CTC_WEIGHT]] = 0.3 加進 [[loss (total)]]
- 放進 [[out (stage1 loss dict)]] 的 `"ctc_loss"` 鍵 → tensorboard `train/ctc_loss`

由 [[log_probs]]、[[ctc_ids]]、[[input_lengths]]、[[ctc_lens]] 算出。
