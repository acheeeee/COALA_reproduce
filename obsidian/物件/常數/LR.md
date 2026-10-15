---
tags: [COALA, 物件, 常數]
類別: 常數
型別: float
---
# LR

← [[物件 總覽]]

**存什麼**：過擬合測試用的學習率。

| 項目 | 內容 |
|---|---|
| 型別 | `float` |
| 值 | `1e-4`（論文未給，LoRA 常見值） |
| 定義於 | [[overfit_test.py]] |

## 用在哪裡
- [[overfit_test.main()]]：`torch.optim.AdamW(params, lr=LR)`

正式訓練的 lr 是命令列參數 `--lr`（預設也是 `1e-4`），存在 [[args]] 裡，不是這個常數。
