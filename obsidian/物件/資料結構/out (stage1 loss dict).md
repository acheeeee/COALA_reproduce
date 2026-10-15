---
tags: [COALA, 物件, 資料結構, loss]
類別: 資料結構
型別: dict
aliases: [out]
---
# out (stage1 loss dict)

← [[物件 總覽]]

**存什麼**：Stage 1 一次 forward 的三個 loss。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[COALAModel.forward_stage1()]]：`dict(loss=total, ctc_loss=ctc_loss, ce_loss=ce_loss)` |

| 鍵 | 物件 | 用途 |
|---|---|---|
| `"loss"` | [[loss (total)]] | `.backward()` |
| `"ctc_loss"` | [[ctc_loss]] | 記錄 |
| `"ce_loss"` | [[ce_loss]] | 記錄 |

## 誰在用
[[train_stage1.main()]]（backward + tensorboard + epoch 累加）、[[validate()]]、[[overfit_test.main()]]、[[model.main()]]

> 變數名 `out` 在別處有別的意思：[[out (encoder 輸出物件)]]、[[out (LM 輸出物件)]]。
