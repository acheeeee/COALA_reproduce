---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: torch.optim.AdamW
---
# opt

← [[物件 總覽]]

**是什麼**：優化器，依梯度更新 [[params]]。

| 地方 | 設定 |
|---|---|
| [[train_stage1.main()]] | `AdamW(params, lr=args.lr, weight_decay=0.01)` |
| [[overfit_test.main()]] | `AdamW(params, lr=LR)`（weight_decay 用 PyTorch 預設 0.01） |

論文沒給 optimizer 和 lr（疑點 #1），AdamW + 1e-4 是自選。

## 每步的固定三連
```python
opt.zero_grad()           # 清掉上一步的梯度（不清會累加）
out["loss"].backward()    # 算梯度
opt.step()                # 更新參數
```

狀態（一階、二階動量）會存進 [[ck (checkpoint dict)]]，續跑時要一起還原。
