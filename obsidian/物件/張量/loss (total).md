---
tags: [COALA, 物件, 張量, loss]
類別: 張量
型別: torch.Tensor
shape: () 純量
dtype: float32
aliases: [loss, total]
---
# loss (total)

← [[物件 總覽]]

**存什麼**：Stage 1 真正拿去 `backward()` 的總 loss。

```python
total = CTC_WEIGHT * ctc_loss + CE_WEIGHT * ce_loss     # 0.3·ctc + 1.0·ce
```

| 項目 | 內容 |
|---|---|
| shape | `()` |
| 定義於 | [[COALAModel.forward_stage1()]]（區域變數名 `total`） |
| 取用 | `out["loss"]`（[[out (stage1 loss dict)]]） |

## 去處
- `out["loss"].backward()`：[[train_stage1.main()]]、[[overfit_test.main()]]、[[model.main()]]
- [[validate()]] 平均後當選模型的依據（val_total）

## 實測

| 設定 | 最佳 val_total |
|---|---|
| stack=4 | 2.0163 |
| **stack=2** | **0.4246**（−79%） |

由 [[ctc_loss]]、[[ce_loss]] 組成。
