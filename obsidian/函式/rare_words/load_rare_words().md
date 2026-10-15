---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: rare_words.py
---
# load_rare_words()

← [[rare_words.py]] · [[函式 總覽]]

**功能**：載入 209,291 個 rare word（set 版）→ 只用於**採樣** E⁻，不用於判定。

```python
def load_rare_words(path=RARE_WORDS_FILE):
    return load_word_list(path)
```

| | 物件 |
|---|---|
| 輸入 | `path`，預設 [[RARE_WORDS_FILE]] |
| 輸出 | `set[str]`，209,291 個 |

## 呼叫了誰
[[load_word_list()]]

## 被誰呼叫
[[load_rare_pool()]]（再排序成 list → [[rare_pool]]）
