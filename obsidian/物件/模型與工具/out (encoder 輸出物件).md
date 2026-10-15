---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: transformers BaseModelOutput
aliases: [out]
---
# out (encoder 輸出物件)

← [[物件 總覽]]

**是什麼**：[[encoder]] 回傳的包裝物件，真正要的東西在 `.last_hidden_state`。

| 項目 | 內容 |
|---|---|
| 型別 | `BaseModelOutput`（像 dict，也能用屬性取值） |
| 定義於 | [[COALAModel.encode_audio()]]、[[check_shape.main()]] |
| 取用 | `out.last_hidden_state` → `(B, 1500, 1280)` → [[x (encoder 輸出)]] |

> 變數名 `out` 在別處有別的意思：[[out (LM 輸出物件)]]、[[out (stage1 loss dict)]]。
