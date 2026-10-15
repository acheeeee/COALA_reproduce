---
tags: [COALA, 函式, 模型層]
種類: main
檔案: modules.py
---
# modules.main()

← [[modules.py]] · [[函式 總覽]]

**功能**：用隨機資料檢查 adapter / CTC head 的 shape、參數量，以及 CTC loss 的量級。

## 步驟
1. `x = torch.randn(2, 1500, 1280)` → [[AudioAdapter]] → [[h (adapter 輸出)]]，印 shape
2. → [[CTCHead]] → `logits`，印 shape
3. 數兩個模組的參數量，對照 5.7M 預算
4. 假的 CTC 資料：`logits (4, 375, 29)`、`targets` 用 `randint(1, 29)`（避開 0 = blank）、`target_lengths = [89, 63, 172, 133]` → [[compute_ctc_loss()]]

## 判斷對不對
CTC loss 應約 10（實測 10.18）。

> ⚠️ 第 4 步寫死了 T = 375（stack=4 時代），和現在預設 stack=2（750）不一致，但這段是獨立的 loss 測試，不影響結果判讀。

`torch.randn`：標準常態分布（平均 0、標準差 1）的隨機數。
