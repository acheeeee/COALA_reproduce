---
tags: [COALA, 物件, 資料結構, 詞表]
類別: 資料結構
型別: set[str]
---
# common_set

← [[物件 總覽]]

**存什麼**：5000 個常見詞。**不在這裡面的字就是 rare word。**

| 項目 | 內容 |
|---|---|
| 型別 | `set[str]`，5000 個 |
| 來源 | [[common_words_5k.txt]]，經 [[load_common_words()]] 讀入 |

## 為什麼用 set
[[find_rare_words()]] 要對每個字問「在不在裡面」。set 是 hash 查表 O(1)；list 要從頭比對 O(n)。

## 誰在用
- [[find_rare_words()]]（第二個參數，⚠️ 是 common 不是 rare）
- [[build_manifest()]]、[[figure1.main()]]、[[biasing.main()]]、[[rare_words.main()]]
