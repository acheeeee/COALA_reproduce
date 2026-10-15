---
tags: [COALA, 物件, 常數, 路徑]
類別: 常數
型別: pathlib.Path
---
# RARE_WORDS_FILE

← [[物件 總覽]]

**存什麼**：全部 rare word 的詞表路徑 → 指向資料檔 [[all_rare_words.txt]]。

| 項目 | 內容 |
|---|---|
| 型別 | `Path` |
| 值 | [[WORDS_DIR]] `/ "all_rare_words.txt"` |
| 定義於 | [[rare_words.py]]（[[biasing.py]] 也 import 它） |

## 用在哪裡
- [[load_rare_words()]] 的預設參數
- [[load_rare_pool()]] 的預設參數 → 產生 [[rare_pool]]

> [!warning] 只拿來**採樣干擾詞 E⁻**，不能拿來判定 E⁺（它缺 OOV 詞，只對得上 2512/2620）。
