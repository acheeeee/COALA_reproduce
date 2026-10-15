---
tags: [COALA, 函式, 資料層, biasing]
種類: main
檔案: biasing.py
---
# biasing.main()

← [[biasing.py]] · [[函式 總覽]]

**功能**：三段驗證 —— 產生器正確且可重現、Stage 1 清單長度對、自己的 E⁺ 和官方一致。

## 1. 產生器 + 可重現性
- 用 `"...mated and intermingled..."` 做 [[positives (E⁺)]]
- [[make_biasing_list()]] 跑三次：seed 42、42、43 → `lst_a`、`lst_b`、`lst_c`
- 檢查：正樣本都在、42 == 42、42 != 43
- [[add_unbiased()]] + [[split_pos_neg()]] → 印 |E⁺|、|E⁻|
- 再用一句**沒有** rare word 的句子，確認 E⁺ = `[<unbiased>]`

## 2. Stage 1 清單
7 個假正樣本 → [[make_stage1_list()]] → 長度 10、實際含 5 個正樣本

## 3. 和官方比對
[[load_official_ref()]]`("test-clean", 500)` → 每句：
- 官方 E⁺ ⊆ 官方清單？（`n_superset_ok`）
- [[find_rare_words()]] 的結果 == 官方 E⁺？（`n_match_ok`）→ **2620/2620**
- `len_dist`（`collections.Counter`）：統計「清單長度 − 正樣本數」，確認官方是固定補 N 個干擾詞

## 用到的物件
[[rare_pool]]、[[common_set]]、[[biasing_list]]、[[full_list (E)]]、[[pos_idx]]、[[neg_idx]]、[[ref (官方 ref dict)]]、[[rng]]
