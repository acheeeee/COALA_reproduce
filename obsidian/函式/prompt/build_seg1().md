---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# build_seg1()

← [[prompt.py]] · [[函式 總覽]]

**功能**：組出 prompt 第 1 段文字（system 訊息 + user turn 開頭）。

```python
def build_seg1(tokenizer):
    return (
        f"{tokenizer.bos_token}system\n{SYSTEM_PROMPT}{tokenizer.eos_token}\n"
        f"{tokenizer.bos_token}user\n"
    )
```

| | 物件 |
|---|---|
| 輸入 | [[tokenizer]]（只用它的 `bos_token` / `eos_token` 字串） |
| 輸出 | [[seg1]]（`str`） |

用到：[[SYSTEM_PROMPT]]

被 [[build_stage1_sample()]]、[[build_gen_prompt()]]、[[prompt.main()]] 呼叫。
