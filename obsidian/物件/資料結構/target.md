---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: str
---
# target

← [[物件 總覽]]

**存什麼**：prompt 的第 4 段文字 —— 正確逐字稿 + 結束符號。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 定義於 | [[build_stage1_sample()]] |

```python
target = target_text + tokenizer.eos_token     # "mister quilter ... gospel<|im_end|>"
```

## 為什麼要加 eos
讓 LM 學會「說完就停」。沒有它，生成時會一直講下去，直到 [[MAX_NEW_TOKENS]]。

→ [[target_ids]]
