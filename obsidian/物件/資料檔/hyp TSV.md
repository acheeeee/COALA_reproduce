---
tags: [COALA, 物件, 資料檔, 評估]
類別: 資料檔
位置: exp/hyp/{split}.{s2|s4}.{mode}.tsv
---
# hyp TSV

← [[物件 總覽]]

**存什麼**：模型對每句話的轉寫，每行 `{utt_id}\t{hyp}`，沒有表頭。

寫入：[[evaluate.main()]]；讀取：[[score.py (官方)]]

> [!bug] 目前的兩個 bug
> - Bug A：[[hyp]] 含換行 → 行數變多（2629 行 vs 2611 句）
> - Bug B：評估時也過濾了 >30 秒的句子 → 少 9 句 → score.py 報 missing
