---
tags: [COALA, 函式, 資料層, CTC]
種類: 函式
檔案: dataset.py
---
# text_to_ctc_ids()

← [[dataset.py]] · [[函式 總覽]]

**功能**：逐字稿 → CTC 字元 id 序列（表外字元直接跳過）。

```python
def text_to_ctc_ids(text):
    return [CTC_CHAR2ID[c] for c in text if c in CTC_CHAR2ID]
```

| | 型別 | 物件 |
|---|---|---|
| 輸入 `text` | `str` | [[text]]（小寫） |
| 輸出 | `list[int]` | [[ctc_ids]] 的 list 版（`id_list`） |

用到：[[CTC_CHAR2ID]]

被 [[LibriSpeechDataset.__getitem__()]] 呼叫。

> 閱讀順序：`for c in text` → `if c in CTC_CHAR2ID` → 取 `CTC_CHAR2ID[c]`。
> 例：`"he hoped"` → `[10, 7, 1, 10, 17, 18, 7, 6]`
