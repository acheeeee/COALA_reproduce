---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: list[str]
aliases: [el, entities]
---
# entity_list

← [[物件 總覽]]

**存什麼**：要寫進 prompt 指令裡、提示 LM「這些詞可能出現」的單字清單。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]` |
| 別名 | `el`（[[train_stage1.py]]、[[evaluate.py]]）、`entities`（[[prompt.main()]]） |

## 三種來源（[[evaluate.main()]] 的 `--mode`）

| mode | entity_list | 意義 |
|---|---|---|
| `nobias` | `[]` | 不給提示 → [[INSTRUCTION_NO_BIAS]] |
| `oracle` | [[positives (E⁺)]] | 只給正確答案（上限） |
| `stage1` | [[make_stage1_list()]] 的 10 個 | 和訓練時相同的條件 |

訓練時一律是 `stage1` 版本（[[make_entity_lists()]]）。

## 去處
[[build_seg2()]] → `", ".join(entity_list)` 塞進 [[INSTRUCTION_TEMPLATE]] → [[seg2]]
