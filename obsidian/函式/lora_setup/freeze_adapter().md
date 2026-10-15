---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# freeze_adapter()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：把某一組 LoRA 的參數全部設成不可訓練。

```python
def freeze_adapter(model, name):
    for n, p in model.named_parameters():
        if f".{name}." in n:
            p.requires_grad = False
```

| 參數 | 例 |
|---|---|
| `model` | [[lm]] |
| `name` | `"asr"` |

靠參數名辨識：LoRA 參數名長這樣 `...q_proj.lora_A.asr.weight`，所以找含 `.asr.` 的。前後的點避免誤中名字裡剛好含 "asr" 的其他參數。

## 用途
Stage 2：`set_active(model, ["asr", "scoring"])` 之後 `freeze_adapter(model, "asr")` → 兩組都參與 forward，但只有 scoring 會被更新（論文 3.4「All previously trained parameters are frozen」）。

被 [[lora_setup.main()]] 呼叫。
