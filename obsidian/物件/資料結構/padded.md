---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: dict
---
# padded

← [[物件 總覽]]

**存什麼**：pad 成同樣長度的 batch 版三件組。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[pad_batch()]] |
| 用在 | [[COALAModel.forward_stage1()]]：`out = self.lm(**padded)` |

| 鍵 | shape | pad 值 |
|---|---|---|
| `inputs_embeds` | `(B, max_tot, 576)` | 0 |
| `labels` | `(B, max_tot)` | -100（[[IGNORE_INDEX]]） |
| `attention_mask` | `(B, max_tot)` | 0 |

`**padded` 把 dict 拆成關鍵字參數：等於 `self.lm(inputs_embeds=..., labels=..., attention_mask=...)`。鍵名剛好和 HF 的參數名一樣，所以可以這樣寫。
