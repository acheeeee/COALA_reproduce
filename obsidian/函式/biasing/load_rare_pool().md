---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: biasing.py
---
# load_rare_pool()

← [[biasing.py]] · [[函式 總覽]]

**功能**：載入 rare word 詞池，**排序成 list**，讓抽樣可重現。

```python
def load_rare_pool(path=RARE_WORDS_FILE):
    return sorted(load_rare_words(path))
```

| | 物件 |
|---|---|
| 輸入 | `path`，預設 [[RARE_WORDS_FILE]] |
| 輸出 | [[rare_pool]]（`list[str]`，209,291 個，字母序） |

## 呼叫了誰
[[load_rare_words()]]

## 被誰呼叫
[[train_stage1.main()]]、[[evaluate.main()]]、[[overfit_test.main()]]、[[model.main()]]、[[biasing.main()]]

> `sorted()` 的關鍵作用：set → 有固定順序的 list。`rng.sample` 對 set 抽樣，結果會隨 hash 順序改變。
