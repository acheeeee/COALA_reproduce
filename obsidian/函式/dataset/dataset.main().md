---
tags: [COALA, 函式, 資料層]
種類: main
檔案: dataset.py
---
# dataset.main()

← [[dataset.py]] · [[函式 總覽]]

**功能**：檢查單筆和一個 batch 的 shape。

## 步驟
1. 建 [[fe]]（[[WHISPER_ID]]）和 [[LibriSpeechDataset]]（dev-clean）
2. `s = ds[0]` → 印 [[input_features]] `(80, 3000)`、[[ctc_ids]] `(89,)`、[[text]]、[[positives (E⁺)]]
3. DataLoader（batch 4）取一個 [[batch]] → 印 `(4, 80, 3000)`、`(4, 172)`、`ctc_lens=[89, 63, 172, 133]`

## 判斷對不對
- `ctc_ids` 第一句 89 = 73 個字母 + 16 個空格 ✓
- batch 的 `ctc_ids` 第二維 = `ctc_lens` 的最大值（172）✓
