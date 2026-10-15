---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: dict
aliases: [b]
---
# batch

← [[物件 總覽]]

**存什麼**：B 句話打包在一起，模型一次 forward 的輸入。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 由誰產生 | [[collate_fn()]]（DataLoader 自動呼叫） |

| 鍵 | 型別 / shape | 物件 |
|---|---|---|
| `utt_id` | `list[str]`，長 B | [[utt_id]] |
| `input_features` | `(B, 80, 3000)` float32 | [[input_features]] |
| `text` | `list[str]` | [[text]] |
| `ctc_ids` | `(B, S_max)` long，pad 0 | [[ctc_ids]] |
| `ctc_lens` | `(B,)` long | [[ctc_lens]] |
| `rare_words` | `list[list[str]]` | [[positives (E⁺)]] |

## 上 GPU
只有三個 tensor 欄位要搬：[[to_device()]] 處理 `input_features`、`ctc_ids`、`ctc_lens`；字串和 list 留在 CPU。

## 去處
[[COALAModel.forward_stage1()]] 的第一個參數。
