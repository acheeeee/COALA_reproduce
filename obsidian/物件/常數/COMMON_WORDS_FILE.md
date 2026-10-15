---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# COMMON_WORDS_FILE

← [[物件 總覽]]

**存什麼**：5000 個常見詞的詞表路徑 → 指向資料檔 [[common_words_5k.txt]]。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | [[WORDS_DIR]] `/ "common_words_5k.txt"` |
| 定義於 | [[rare_words.py]] |

## 用在哪裡
- [[load_common_words()]] 的預設參數 → 產生 [[common_set]]

> [!important] 判定 rare word 用的是**它的補集**，不是 [[RARE_WORDS_FILE]]。
