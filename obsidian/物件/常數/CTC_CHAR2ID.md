---
tags: [COALA, 物件, 常數]
類別: 常數
型別: dict[str, int]
---
# CTC_CHAR2ID

← [[物件 總覽]]

**存什麼**：字元 → CTC id 的對照表。

| 項目 | 內容 |
|---|---|
| 型別 | `dict[str, int]`，28 個鍵 |
| 長相 | `{' ': 1, "'": 2, 'a': 3, …, 'z': 28}` |
| 定義於 | [[dataset.py]] |

```python
CTC_CHAR2ID = {c: i + 1 for i, c in enumerate(CTC_CHARS)}   # +1：把 0 讓給 blank
```

## 用在哪裡
- [[text_to_ctc_ids()]]：`[CTC_CHAR2ID[c] for c in text if c in CTC_CHAR2ID]` → 產生 [[ctc_ids]]

由 [[CTC_CHARS]] 推出；id 0 保留給 [[CTC_BLANK_ID]]。
