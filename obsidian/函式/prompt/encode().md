---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# encode()

← [[prompt.py]] · [[函式 總覽]]

**功能**：文字 → token id tensor，**不**自動加 special token，並放到指定裝置。

```python
def encode(text, tokenizer, device):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor(ids, dtype=torch.long, device=device)
```

| | 型別 | 物件 |
|---|---|---|
| 輸入 `text` | `str` | [[seg1]] / [[seg2]] / [[target]] |
| 輸入 `tokenizer` | | [[tokenizer]] |
| 輸入 `device` | `torch.device` | [[dev]] |
| 輸出 | `(n,)` long | [[seg1_ids]] / [[seg2_ids]] / [[target_ids]] |

`tokenizer(text)` 回傳類似 dict 的物件（含 `input_ids`、`attention_mask`），這裡只取 `input_ids`。

被 [[build_stage1_sample()]]、[[build_gen_prompt()]] 呼叫。
