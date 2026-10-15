---
tags: [COALA, 物件, 模型與工具, LoRA]
類別: 模型與工具
型別: peft.LoraConfig
---
# lora_config

← [[物件 總覽]]

**是什麼**：描述「LoRA 要怎麼掛」的設定物件。本身不含權重。

| 項目 | 內容 |
|---|---|
| 型別 | `peft.LoraConfig` |
| 由誰產生 | [[make_lora_config()]]（每次呼叫產生一個新物件） |

| 欄位 | 值 | 物件 |
|---|---|---|
| `r` | 16 | [[LORA_RANK]] |
| `lora_alpha` | 32 | [[LORA_ALPHA]] |
| `lora_dropout` | 0.05 | [[LORA_DROPOUT]] |
| `target_modules` | `["q_proj", "v_proj"]` | [[TARGET_MODULES]] |
| `bias` | `"none"` | 不訓練原本的 bias |
| `task_type` | `"CAUSAL_LM"` | |

## 用在
- `get_peft_model(lm, config, adapter_name="asr")`
- `lm.add_adapter("scoring", config)`

兩組 adapter 用**同樣設定**，所以參數量相同（各 921,600）→ 光數參數量分不出「現在是哪組在訓練」，[[which_gets_grad()]] 才要用 backward 驗證。
