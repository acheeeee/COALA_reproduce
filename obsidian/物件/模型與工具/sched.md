---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: torch.optim.lr_scheduler.LambdaLR
---
# sched

← [[物件 總覽]]

**是什麼**：學習率排程器 —— 先線性升溫，再線性降到 0。

| 項目 | 內容 |
|---|---|
| 建立 | `get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)` |
| warmup 步數 | `int(total_steps × 0.03)` |
| total_steps | `len(train_dl) × epochs` = 7134 × 10 = 71,340（train-clean-100） |
| 定義於 | [[train_stage1.main()]] |

```
lr
1e-4 ┤   ╱‾‾‾‾‾‾‾‾‾‾‾╲___
     │  ╱                 ‾‾‾‾╲___
   0 ┼─┴──────────────────────────╲── step
      3%                        100%
```

每個 **batch** 呼叫一次 `sched.step()`（不是每個 epoch）。當前 lr 用 `sched.get_last_lr()[0]` 取得，記到 tensorboard 和 [[epoch_summary.tsv]]。
