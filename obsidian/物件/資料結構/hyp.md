---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: str
---
# hyp

← [[物件 總覽]]

**存什麼**：模型對一句話的轉寫預測（hypothesis）。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 由誰產生 | [[transcribe_one()]]：decode [[out_ids]] |
| 去處 | [[evaluate.main()]] 寫成 [[hyp TSV]] 的一行：`f"{utt_id}\t{hyp}\n"` |

> [!bug] [[evaluate.py]] Bug A
> 目前只做 `.strip()`，字串中間的 `\n` 會讓一句變成兩行（hyp 檔 2629 行 vs 2611 句）。修法：`" ".join(text.split())`。
