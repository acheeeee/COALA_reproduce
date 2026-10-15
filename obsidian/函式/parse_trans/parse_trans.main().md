---
tags: [COALA, 函式, 資料層]
種類: main
檔案: parse_trans.py
---
# parse_trans.main()

← [[parse_trans.py]] · [[函式 總覽]]

**功能**：示範解析一個 trans.txt，印出句數、前 3 句、最長與最短句。

## 步驟
1. [[parse_trans_file()]]`(`[[TRANS]]`)` → [[items (trans 解析結果)]]
2. 印前 3 句的 [[utt_id]]、字數、前 50 字元
3. 建 `lengths = [(字數, utt_id), ...]`，用 `max/min(..., key=lambda x: x[0])` 找最長、最短

## 用到的物件
[[TRANS]]、[[items (trans 解析結果)]]、`lengths`（`list[tuple[int, str]]`）

> `key=lambda x: x[0]` = 「比較時只看 tuple 的第 0 個元素（字數）」。
