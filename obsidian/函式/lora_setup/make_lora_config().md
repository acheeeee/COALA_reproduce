---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# make_lora_config()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：產生一份 LoRA 設定。每次呼叫都回傳**新的**物件。

```python
def make_lora_config():
    return LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
```

| 輸入 | 輸出 |
|---|---|
| 無 | [[lora_config]] |

用到：[[LORA_RANK]]、[[LORA_ALPHA]]、[[LORA_DROPOUT]]、[[TARGET_MODULES]]

## 被誰呼叫
[[COALAModel.__init__()]]（兩次：asr、scoring）、[[lora_setup.main()]]

> 寫成函式而不是一個全域 config 物件：兩組 adapter 各拿一份獨立的 config，避免共用同一個物件時被 peft 內部修改互相影響。
