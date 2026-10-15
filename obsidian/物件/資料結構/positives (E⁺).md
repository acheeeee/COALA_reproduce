---
tags: [COALA, 物件, 資料結構, biasing]
類別: 資料結構
型別: list[str]
aliases: [positives, rare_words, rw, pos, mine, E+]
---
# positives (E⁺)

← [[物件 總覽]]

**存什麼**：這句話裡真的出現的 rare word = 論文的**正樣本 E⁺**。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]`，去重 + 字母排序 |
| 可能是空的 | 是 —— 這句沒有 rare word（train-clean-100 有 7.4%、dev-clean 23.68%） |
| 別名 | manifest 的 `rare_words` 欄、`batch["rare_words"][i]`、`rw`、`pos`、`mine` |

## 例
`"the air and the earth are curiously mated and intermingled ..."` → `["intermingled", "mated"]`

## 從哪來
[[find_rare_words()]]（[[text]] 裡不在 [[common_set]] 的字）→ 存進 [[manifest TSV]] → [[sample (Dataset 單筆)]]`["rare_words"]`

## 誰在用

| 誰 | 用途 |
|---|---|
| [[make_biasing_list()]] / [[make_stage1_list()]] | 放進 biasing list |
| [[split_pos_neg()]] | 決定 [[pos_idx]]；**空的話** → [[UNBIASED]] 變成唯一正樣本 |
| [[evaluate.main()]] `--mode oracle` | 直接當 [[entity_list]] |

> [!important] 判定時用 common 的補集
> 用 [[all_rare_words.txt]] 判定只有 2512/2620 對；用 [[common_set]] 取補集 → 2620/2620 和官方一致。
