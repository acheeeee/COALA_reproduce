---
tags: [COALA, 函式, 資料層]
種類: main
檔案: figure1.py
---
# figure1.main()

← [[figure1.py]] · [[函式 總覽]]

**功能**：統計 dev-clean 每句含幾個 rare word，重現論文 Figure 1 的分布，並檢查論文「99% 的句子少於 10 個」的說法。

## 步驟
1. [[load_common_words()]] → [[common_set]]
2. 走訪 [[SPLIT_DIR]] 下所有 [[trans.txt]]，每句：
   - `n = len(find_rare_words(text, common_set))`
   - `counts[bucket_of(n)] += 1`
   - 追蹤最大值 `max_n` / `max_utt`、少於 10 個的數量 `n_lt_10`
3. 照 [[BUCKET_ORDER]] 印表

## 內部物件

| 名稱 | 型別 | 角色 |
|---|---|---|
| `counts` | `dict[str, int]` | 四個桶的計數 |
| `total` | `int` | 總句數 |
| `max_n`, `max_utt` | `int`, `str` | rare word 最多的那句 |
| `n_lt_10` | `int` | 少於 10 個的句數 |

## 呼叫了誰
[[parse_trans_file()]]、[[find_rare_words()]]、[[bucket_of()]]

> 這支直接讀 trans.txt，沒有用 manifest（寫在 manifest 之前）。
