---
tags: [COALA, 物件, 常數, prompt, Stage2]
類別: 常數
型別: str
---
# STAGE2_INSTRUCTION

← [[物件 總覽]]

**存什麼**：Stage 2 的題目句 —— 請 LM 判斷「接下來這個詞有沒有出現在語音裡」。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"Please determine if the text appears in the input speech. Text:"` |
| token 數 | 13 |
| 定義於 | [[prompt.py]] |

## 來源
論文圖 2(a) 只露出 *"...Please determine if the text appears in the input speech"*，前後文字是自訂的（疑點 #7 的延伸）。

## 為什麼結尾是 `Text:`（沒有空格）
後面接的 entity 是用 `" " + e` encode 的，空格放在 entity 那邊 → 和整句一起 tokenize 的結果一致（斷詞邊界檢查全 ✅）。

## 用在哪裡
- [[build_stage2_sample()]]：encode → 查表 → 放在 [[prefix_embeds]] 的最後一段
- [[prompt.main()]]：算 n_inst、做斷詞邊界檢查

> 和 Stage 1 的 [[INSTRUCTION_TEMPLATE]] 不同：那個是「轉寫，可能出現這些詞」，這個是「判斷這**一個**詞在不在」。
