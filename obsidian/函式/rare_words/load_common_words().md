---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: rare_words.py
---
# load_common_words()

← [[rare_words.py]] · [[函式 總覽]]

**功能**：載入 5000 個常見詞 → 用來**判定** rare word。

```python
def load_common_words(path=COMMON_WORDS_FILE):
    return load_word_list(path)
```

| | 物件 |
|---|---|
| 輸入 | `path`，預設 [[COMMON_WORDS_FILE]] |
| 輸出 | [[common_set]]（`set[str]`，5000 個） |

## 呼叫了誰
[[load_word_list()]]

## 被誰呼叫
[[build_manifest.main()]]、[[figure1.main()]]、[[biasing.main()]]、[[rare_words.main()]]

> 只是 [[load_word_list()]] 換個名字 + 給預設路徑。好處是呼叫端讀起來就知道「這是 common 詞表」。
