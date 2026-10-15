---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: str
aliases: [target_text]
---
# text

← [[物件 總覽]]

**存什麼**：一句話的正確逐字稿。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 格式 | **全小寫**、只有 a–z、空格、撇號（[[trans.txt]] 原本是全大寫，[[build_manifest()]] 轉小寫） |
| 例 | `"mister quilter is the apostle of the middle classes"` |
| 別名 | `target_text`（[[build_stage1_sample()]] 的參數） |

## 流經的地方

| 誰 | 拿來做什麼 |
|---|---|
| [[find_rare_words()]] | 切詞找出 [[positives (E⁺)]] |
| [[text_to_ctc_ids()]] | 逐字元轉成 [[ctc_ids]] |
| [[build_stage1_sample()]] | 加上 eos 變成 [[target]] → [[target_ids]] |
| [[evaluate.main()]] | 前 3 句印出 REF 和 [[hyp]] 對照 |
