---
tags: [COALA, 物件, 常數]
類別: 常數
型別: str
---
# LM_ID

← [[物件 總覽]]

**存什麼**：backbone LM（SmolLM2）在 HuggingFace Hub 上的名稱。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"HuggingFaceTB/SmolLM2-135M-Instruct"` |
| 定義於 | [[lora_setup.py]]（主要）、[[dataset.py]]（定義了但沒用到）、[[check_shape.py]] |
| 別名 | [[prompt.main()]] 裡的區域變數 `MID`（同一個字串） |

## 用在哪裡

| 誰用 | 拿來做什麼 |
|---|---|
| [[COALAModel.__init__()]] | 載入 [[lm]] 與 [[tokenizer]]（`from lora_setup import LM_ID`） |
| [[lora_setup.main()]] | 單獨測 LoRA 時載入 LM |
| [[check_shape.main()]] | 讀出 [[emb (embedding 權重)]] 確認 `D = 576` |
