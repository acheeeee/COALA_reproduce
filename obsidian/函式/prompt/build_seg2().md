---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# build_seg2()

← [[prompt.py]] · [[函式 總覽]]

**功能**：組出 prompt 第 3 段文字（指令 + entity 清單 + 開啟 assistant turn）。

```python
def build_seg2(entity_list, tokenizer):
```

| | 物件 |
|---|---|
| 輸入 `entity_list` | [[entity_list]]（可以是空的） |
| 輸入 `tokenizer` | [[tokenizer]] |
| 輸出 | [[seg2]]（`str`） |

## 內部物件
[[instruction]]：
```python
if entity_list:
    instruction = INSTRUCTION_TEMPLATE.format(entities=", ".join(entity_list))
else:
    instruction = INSTRUCTION_NO_BIAS
return f"{instruction}{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n"
```

用到：[[INSTRUCTION_TEMPLATE]]、[[INSTRUCTION_NO_BIAS]]

- `", ".join(list)`：用「逗號 + 空格」把 list 接成一個字串
- `.format(entities=...)`：把字串填進模板的 `{entities}`

被 [[build_stage1_sample()]]、[[build_gen_prompt()]]、[[prompt.main()]] 呼叫。

> 函式第一行在 if 之前多算了一次 `instruction`，馬上被覆蓋 → 可刪的冗餘行。
