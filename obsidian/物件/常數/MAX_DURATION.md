---
tags: [COALA, 物件, 常數]
類別: 常數
型別: float
---
# MAX_DURATION

← [[物件 總覽]]

**存什麼**：Dataset 允許的最長音檔秒數，超過就整句跳過。

| 項目 | 內容 |
|---|---|
| 型別 | `float` |
| 值 | `30.0`（單位：秒） |
| 定義於 | [[dataset.py]] |

## 為什麼是 30
Whisper 的 [[fe]] 會把所有音檔 pad／截斷成固定 30 秒（→ [[input_features]] 固定 `(80, 3000)`）。超過 30 秒的部分會被截掉，但逐字稿還是完整的 → 等於教模型輸出它沒聽到的字。

## 用在哪裡
- [[LibriSpeechDataset.__init__()]] 的參數 `max_duration` 預設值

> [!warning] 評估時不該用它
> 訓練時過濾是對的，但 [[evaluate.main()]] 也沿用了預設值 → test-clean 少測 9 句，`score.py` 報 missing。修法：評估時傳 `max_duration=1e9`。見 [[evaluate.py]] 的 Bug B。
