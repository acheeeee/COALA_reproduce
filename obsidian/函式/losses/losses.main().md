---
tags: [COALA, 函式, 模型層, loss]
種類: main
檔案: losses.py
---
# losses.main()

← [[losses.py]] · [[函式 總覽]]

**功能**：用論文圖 2(b) 的 5 個手寫分數，驗證四個 loss 的數值、穩定性、梯度方向。**完全不需要模型。**

## 測試資料

| i | entity | sᵢ | E⁺？ |
|---|---|---|---|
| 0 | `<unbiased>` | −2.0 | |
| 1 | Watson | 3.0 | ✅ |
| 2 | solved | 1.0 | ✅ |
| 3 | survey | −1.0 | |
| 4 | collect | −3.0 | |

`s` 設 `requires_grad=True`（測試 4 要看梯度）；`pos`、`pos1`（只有 `<unbiased>`）是 [[pos (E⁺ 遮罩)]]。

## 5 項測試與實測結果（2026-10-18）

| # | 測什麼 | 結果 |
|---|---|---|
| 1 | 手算答案 | disc 1.150889、mpd 0.106171、dpd 0.343850 ✅ |
| 2 | 單一正樣本時 disc = mpd | 都是 5.150889 ✅（E⁺ 只有一個時兩個公式的分母相同） |
| 3 | 極端分數 ±300 | 100.0 / 0.0 / 0.0，全部有限值 ✅ |
| 4 | 梯度 | disc 的 Watson **+0.360**（正樣本被往下推）；mpd、dpd 兩個正樣本都是負梯度 |
| 5 | Bias-Loss（`logp = [−5, −2, −3, −8, −9]`） | 1.535800、4.635800 ✅ |

## 測試 3 為什麼是陷阱
照公式字面寫 `log(exp(s) / exp(s).sum())`：float32 的 exp(x) 在 x > 88.7 就溢位 → `inf / inf = nan`。`logsumexp` 內部先減掉最大值，指數永遠 ≤ 0。

## 呼叫了誰
[[disc_loss()]]、[[mpd_loss()]]、[[dpd_loss()]]、[[bias_loss()]]、[[check()]]
