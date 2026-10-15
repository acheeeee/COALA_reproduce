---
tags: [COALA, 物件, 常數, prompt]
類別: 常數
型別: str
---
# INSTRUCTION_TEMPLATE

← [[物件 總覽]]

**存什麼**：Stage 1 有給 entity 時的指令模板，`{entities}` 是佔位符。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"Transcribe the speech. The following entities may appear: {entities}."` |
| 定義於 | [[prompt.py]] |

## 用在哪裡
- [[build_seg2()]]：`INSTRUCTION_TEMPLATE.format(entities=", ".join(entity_list))` → [[instruction]] → [[seg2]]

[[entity_list]] 為空時改用 [[INSTRUCTION_NO_BIAS]]。
