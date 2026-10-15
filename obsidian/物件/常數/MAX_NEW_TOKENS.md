---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# MAX_NEW_TOKENS

← [[物件 總覽]]

**存什麼**：生成轉寫時最多產生幾個 token。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `200` |
| 定義於 | [[evaluate.py]] |

## 用在哪裡
- [[transcribe_one()]]：`model.lm.generate(..., max_new_tokens=MAX_NEW_TOKENS)`

30 秒內的 LibriSpeech 句子 token 數遠少於 200，正常情況下會先遇到 `<|im_end|>` 停下。若輸出被截在 200，代表模型沒學會停（例如重複迴圈）。
