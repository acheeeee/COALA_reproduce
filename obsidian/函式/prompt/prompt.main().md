---
tags: [COALA, 函式, 模型層, prompt]
種類: main
檔案: prompt.py
---
# prompt.main()

← [[prompt.py]] · [[函式 總覽]]

**功能**：用假的音訊 token 檢查 [[build_stage1_sample()]] 的三個 shape 和 labels 遮罩。

## 步驟
1. 載入 [[tokenizer]]（`tok`）、LM、[[embed_layer]]（`embed`）—— 用區域變數 `MID`（= [[LM_ID]]）
2. `audio = torch.randn(89, 576)`（假裝 CTC 留下 89 個位置）
3. 10 個 entity + 逐字稿 → [[build_stage1_sample()]]
4. 印三個 shape；數 `labels != -100` 的位置數，對照 target 的 token 數
5. 印出 [[seg1]]、[[seg2]] 原文和文字 token 總數

## 判斷對不對（實測）

| 項目 | 值 |
|---|---|
| 三個 shape | `(160, 576)`、`(160,)`、`(160,)` → 第一維必須相同 |
| 算 loss 的位置 | 12 = target token 數 12 ✓ |
| 文字 token 總數 | 59（seg1 15 + seg2 44） |
| 長度組成 | 59 + 89 + 12 = 160 |
