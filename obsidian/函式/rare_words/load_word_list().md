---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: rare_words.py
---
# load_word_list()

← [[rare_words.py]] · [[函式 總覽]]

**功能**：讀「一行一個詞」的文字檔，回傳 set。另外兩個 load 函式的共同底層。

```python
def load_word_list(path):
```

| | 型別 | 內容 |
|---|---|---|
| 輸入 `path` | `Path` | 詞表檔 |
| 輸出 | `set[str]` | 去掉空白行的所有詞 |

## 內部物件
`words`（`set`，累積結果）、`word`（`line.strip()` 後的一個詞）

## 被誰呼叫
[[load_common_words()]]、[[load_rare_words()]]

```python
def load_word_list(path):
    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                words.add(word)
    return words
```
