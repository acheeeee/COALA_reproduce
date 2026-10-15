---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# set_active()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：切換要啟用哪組 LoRA；可以只啟用一組，或同時啟用多組。

```python
def set_active(model, names):
    if isinstance(names, str):
        model.set_adapter(names)
    else:
        model.base_model.set_adapter(list(names))
```

| 參數 | 型別 | 例 |
|---|---|---|
| `model` | `PeftModel` | [[lm]] |
| `names` | `str` 或 `list[str]` | `"asr"`、`["asr", "scoring"]` |

## 為什麼要分兩種情況
`PeftModel.set_adapter` 只吃單一字串；要同時啟用多組，得呼叫底層 `LoraModel`（`model.base_model`）的 `set_adapter`。

## 副作用
`set_adapter` 會把**被啟用的那組設成可訓練**、其他組凍結。所以 Stage 2「兩組都啟用、只訓練 scoring」需要再呼叫 [[freeze_adapter()]] 把 asr 凍住。

被 [[lora_setup.main()]] 呼叫；Stage 2 會用到。
