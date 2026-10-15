---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: exp/{實驗名}/best.pt、epoch{N}.pt
---
# checkpoint .pt

← [[物件 總覽]]

**存什麼**：訓練狀態的存檔，內容是 [[ck (checkpoint dict)]]。

| 檔名 | 何時寫 |
|---|---|
| `best.pt` | val_total 創新低時（每 2000 步和每個 epoch 結束都會檢查） |
| `epoch{N}.pt` | 每個 epoch 結束 |

## 已有的實驗

| 目錄 | 設定 |
|---|---|
| `exp/stage1_tc100/` | stack=4 |
| `exp/stage1_tc100_s2/` | **stack=2** |

寫入：[[save_ckpt()]]；讀取：[[train_stage1.main()]]（`--resume`）、[[evaluate.main()]]（`--ckpt`）

在 `.gitignore` 裡（`*.pt`），不進 git。
