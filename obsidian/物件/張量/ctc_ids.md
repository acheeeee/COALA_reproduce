---
tags: [COALA, 物件, 張量, CTC]
類別: 張量
型別: torch.Tensor
shape: (n_chars,) / (B, S_max)
dtype: int64
aliases: [targets, id_list]
---
# ctc_ids

← [[物件 總覽]]

**存什麼**：逐字稿轉成的 CTC 字元 id 序列，CTC loss 的「正確答案」。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor`（`id_list` 是轉 tensor 前的 `list[int]`） |
| shape | 單筆 `(n_chars,)`；batch `(B, S_max)`，S_max = 這個 batch 最長那句的字元數 |
| dtype | `torch.long`（int64） |
| 值域 | 1 ~ 28（見 [[CTC_CHAR2ID]]）；batch 版的 pad 位置填 0 |
| 別名 | `targets`（[[compute_ctc_loss()]] 的參數名）、`id_list`（`__getitem__` 內） |

## 例子
`"he hoped"` → `[10, 7, 1, 10, 17, 18, 7, 6]`（h=10, e=7, 空格=1 …）

## 從哪來 → 到哪去
- 產生：[[text_to_ctc_ids()]] → [[LibriSpeechDataset.__getitem__()]] 轉成 tensor → [[collate_fn()]] pad 成 `(B, S_max)`
- 使用：[[compute_ctc_loss()]]，必須搭配 [[ctc_lens]] 才知道每句真正多長

> [!note] pad 填 0 剛好等於 blank id，不會出事
> 因為 `F.ctc_loss` 只讀每列前 `ctc_lens[i]` 個，後面填什麼都不會被看到。
