---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: str
---
# instruction

← [[物件 總覽]]

**存什麼**：填好 entity 的指令句。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 定義於 | [[build_seg2()]] 的區域變數 |

```python
if entity_list:
    instruction = INSTRUCTION_TEMPLATE.format(entities=", ".join(entity_list))
else:
    instruction = INSTRUCTION_NO_BIAS
```
→ [[INSTRUCTION_TEMPLATE]] / [[INSTRUCTION_NO_BIAS]]

> [!note] 小冗餘
> [[build_seg2()]] 在 if 之前多算了一次 `INSTRUCTION_TEMPLATE.format(...)`，結果馬上被 if/else 覆蓋。功能正確，只是那一行可以刪。

> [!question] Stage 2 會需要不同的指令
> 論文圖 2(a) 露出的是 "...Please determine if the text appears in the input speech"（判斷**一個**詞在不在）。
