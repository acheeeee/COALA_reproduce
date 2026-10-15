---
tags: [COALA, 物件, 常數]
類別: 常數
型別: str
---
# UNBIASED

← [[物件 總覽]]

**存什麼**：論文的特殊 entity e₀，代表「這句裡沒有任何清單上的 entity」。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"<unbiased>"` |
| 定義於 | [[biasing.py]] |
| 對應論文 | Sec. 2.1 的 e₀ |

## 身分是條件性的
- 句中**有** rare word → 它屬於 [[negatives (E⁻)]]
- 句中**沒有** rare word → 它是唯一的 [[positives (E⁺)]]

## 用在哪裡
- [[add_unbiased()]]：插在清單最前面 → [[full_list (E)]] 的 index 0
- [[split_pos_neg()]]：無正樣本時用 `full_list.index(UNBIASED)` 找出它

## 之後還會用到（尚未實作）
- Stage 2：當作一個普通候選送進 LM 打分
- BTI：推論時用它的分數 s₀ 當**門檻**，分數比它低的候選丟掉

> [!question] 還沒決定
> 它要用 tokenizer 切成普通 token（`<`, `un`, `biased`, `>` 之類），還是註冊成新的 special token？論文沒寫。
