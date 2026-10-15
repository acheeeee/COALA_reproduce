---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: transformers CausalLMOutputWithPast
aliases: [out]
---
# out (LM 輸出物件)

← [[物件 總覽]]

**是什麼**：[[lm]] forward 回傳的包裝物件。

| 項目 | 內容 |
|---|---|
| 型別 | `CausalLMOutputWithPast` |
| 定義於 | [[COALAModel.forward_stage1()]]：`out = self.lm(**padded)` |

| 屬性 | shape | 用途 |
|---|---|---|
| `.loss` | `()` | 有給 [[labels]] 才會有 → [[ce_loss]] |
| `.logits` | `(B, seq, 49152)` | [[logits (LM 輸出)]] |
| `.past_key_values` | 每層的 K、V | **KV cache**（Stage 2 重用共同前綴的關鍵） |
| `.hidden_states` | 31 個 `(B, seq, 576)` | 要傳 `output_hidden_states=True` 才有；Stage 2 取 `[-1]` → [[h (projector 輸入)]] |

> 變數名 `out` 在別處有別的意思：[[out (encoder 輸出物件)]]、[[out (stage1 loss dict)]]。
