---
tags: [COALA, 物件, 常數, prompt]
類別: 常數
型別: str
---
# SYSTEM_PROMPT

← [[物件 總覽]]

**存什麼**：chat template 的 system 訊息內容。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"You are a speech recognition assistant."` |
| 定義於 | [[prompt.py]] |

## 用在哪裡
- [[build_seg1()]] → 組進 [[seg1]]

取代 SmolLM2 預設的 `"You are a helpful AI assistant named SmolLM..."`（17 token）。論文沒給完整 prompt 文字，這是自訂的（疑點 #7）。
