---
tags: [COALA, 物件, 張量, Stage2, loss]
類別: 張量
型別: torch.Tensor
shape: (K,)
dtype: bool
aliases: [pos, pos1, pos_log]
---
# pos (E⁺ 遮罩)

← [[物件 總覽]]

**存什麼**：K 個候選裡哪些是正樣本 E⁺（True），其餘是 E⁻（False）。

| 項目 | 內容 |
|---|---|
| shape | `(K,)` |
| dtype | `torch.bool` |
| 用在 | [[disc_loss()]]、[[mpd_loss()]]、[[dpd_loss()]]、[[bias_loss()]] |

## 用法
```python
s[pos]     # 正樣本的分數
s[~pos]    # 負樣本的分數（~ 是 bool 的 NOT）
```

## 怎麼從索引轉過來（Stage 2 會用到）
[[split_pos_neg()]] 回傳的是 [[pos_idx]]（`list[int]`），要轉成遮罩：
```python
pos = torch.zeros(K, dtype=torch.bool)
pos[pos_idx] = True
```

## 兩個保證
- **E⁺ 永遠不空**：沒有 rare word 時 `<unbiased>` 是唯一正樣本 → `.mean()` 不會除以 0
- **E⁻ 永遠不空**：M = 120 個候選裡一定有干擾詞

## 測試用的版本（[[losses.main()]]）

| 名稱 | 值 |
|---|---|
| `pos` | `[F, T, T, F, F]`（Watson、solved） |
| `pos1` | `[T, F, F, F, F]`（只有 `<unbiased>`） |
| `pos_log` | [[bias_loss()]] 裡的複本，index 0 改成 False |
