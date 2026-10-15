---
tags: [COALA, 函式, 模型層, Stage2]
種類: main
檔案: projector.py
---
# projector.main()

← [[projector.py]] · [[函式 總覽]]

**功能**：用假資料做 5 項檢查。

## 測試資料

| 物件 | 怎麼造 |
|---|---|
| [[h (projector 輸入)]] | `torch.randn(121, 7, 576)` |
| [[lens (entity 長度)]] | `torch.randint(1, 8, (121,))` → 每個 entity 1~7 個 token |
| [[mask (projector)]] | `(torch.arange(L)[None, :] < lens[:, None]).float()` |

## 5 項檢查與實測結果（2026-10-15）

| # | 檢查 | 做法 | 結果 |
|---|---|---|---|
| 1 | 參數量 | 對照 `D′(D+2)+1` | 332,929 ✅ |
| 2 | 輸出 shape | | `(121,)` ✅ |
| 3 | pad 不影響分數 | pad 位置填 999 再算一次 | True ✅ |
| 4 | 與逐個計算一致 | 每個 entity 只取前 lens[i] 個、不 pad 單獨算 → `s_ref` | True ✅ |
| 5 | 梯度 | `s.sum().backward()` | 全部參數有梯度 ✅ |

檢查 3 證明 mask 生效；檢查 4 證明分母是 Lᵢ 而不是 L。

> ⚠️ 原始碼的註解「檢查 1」「檢查 2」標籤位置和實際程式順序有點錯開，不影響功能。
