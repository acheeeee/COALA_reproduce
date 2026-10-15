---
tags: [COALA, 物件, 常數, prompt]
類別: 常數
型別: str
---
# INSTRUCTION_NO_BIAS

← [[物件 總覽]]

**存什麼**：沒有任何 entity 時的指令。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"Transcribe the speech."` |
| 定義於 | [[prompt.py]] |

## 用在哪裡
- [[build_seg2()]]：`entity_list` 為空 → 用它
- 實際觸發點：[[evaluate.main()]] 的 `--mode nobias`（`el = []`）

避免出現 `"The following entities may appear: ."` 這種空清單的怪句子。
