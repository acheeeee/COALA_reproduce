---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: rare_words.py
---
# find_rare_words()

← [[rare_words.py]] · [[函式 總覽]]

**功能**：找出一句話裡的所有 rare word = **E⁺**。

```python
def find_rare_words(text, common_set):
```

## 輸入 → 輸出

| | 名稱 | 型別 | 物件 |
|---|---|---|---|
| 輸入 | `text` | `str` | [[text]]（大小寫皆可，內部會 `.lower()`） |
| 輸入 | `common_set` | `set[str]` | [[common_set]] ⚠️ 是 common 不是 rare |
| 輸出 | | `list[str]` | [[positives (E⁺)]]，去重 + 字母排序 |

## 內部物件
`found`（`set`，自動去重）、`word`（`text.lower().split()` 的每個字）

```python
def find_rare_words(text, common_set):
    found = set()
    for word in text.lower().split():
        if word not in common_set:
            found.add(word)
    return sorted(found)
```

## 被誰呼叫
[[build_manifest()]]（存進 manifest）、[[figure1.main()]]、[[biasing.main()]]、[[rare_words.main()]]

## 驗證
和官方 [[官方 ref TSV]] 的 E⁺ 比對：**2620/2620** 一致（[[biasing.main()]]）。

> [!tip] 為什麼回傳 sorted list 而不是 set
> set 沒有固定順序，寫進 manifest 時每次可能不同；排序後輸出可重現，也方便和官方答案逐一比對。
