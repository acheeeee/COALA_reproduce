---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: torch.utils.tensorboard.SummaryWriter
---
# writer

← [[物件 總覽]]

**是什麼**：tensorboard 紀錄器，把 loss 曲線寫成 [[tensorboard events]] 檔。

| 項目 | 內容 |
|---|---|
| 建立 | `SummaryWriter(str(out_dir))`，例 `exp/stage1_tc100` |
| 定義於 | [[train_stage1.main()]] |

## 寫了哪些 tag

| tag | 頻率 |
|---|---|
| `train/total_loss`、`train/ctc_loss`、`train/ce_loss`、`train/lr` | 每 50 步 |
| `val/total_loss`、`val/ctc_loss`、`val/ce_loss` | 每 2000 步 |
| `epoch/train_*`、`epoch/val_*` | 每個 epoch（x 軸是 epoch 不是 step） |

用法：`writer.add_scalar(tag, value, step)`。之後可以用 [[dump_tb.py]] 讀回來。
