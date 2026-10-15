---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: torch.utils.data.DataLoader
aliases: [dl, train_dl, val_dl]
---
# DataLoader (dl)

← [[物件 總覽]]

**是什麼**：把 Dataset 一筆一筆取出、打包成 [[batch]] 的迭代器。

| 變數 | 設定 | 在哪 |
|---|---|---|
| `train_dl` | batch 4、`shuffle=True`、8 workers、`pin_memory`、**`drop_last=True`** | [[train_stage1.main()]] |
| `val_dl` | batch 4、`shuffle=False`、4 workers | [[train_stage1.main()]] |
| `dl` | batch 2、`shuffle=False` | [[model.main()]]、[[overfit_test.main()]] |
| `dl` | batch 4 | [[dataset.main()]] |

全部都設 `collate_fn=collate_fn` → [[collate_fn()]]。

## 參數意義
- `num_workers=8`：開 8 個子行程平行讀音檔 + 算 log-Mel（CPU 工作），GPU 不用等
- `pin_memory=True`：放在鎖頁記憶體，搬上 GPU 更快（搭配 [[to_device()]] 的 `non_blocking=True`）
- `drop_last=True`：丟掉最後湊不滿 4 句的 batch → 每步 shape 一致；7134 步/epoch 由此而來（[[STEPS_PER_EPOCH]]）

`len(train_dl)` = 一個 epoch 的步數。
