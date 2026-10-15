---
tags: [COALA, 物件, 常數, LoRA]
類別: 常數
型別: list[str]
---
# TARGET_MODULES

← [[物件 總覽]]

**存什麼**：LoRA 要掛在 LM 的哪些 Linear 層。

| 項目 | 內容 |
|---|---|
| 型別 | `list[str]` |
| 值 | `["q_proj", "v_proj"]` |
| 定義於 | [[lora_setup.py]] |

## 為什麼選這兩個
LoRA 原論文 Table 5：固定參數預算下，同時調 Wq + Wv 優於只調 Wq 或四個全調。

SmolLM2 是 GQA（9 個 query head、3 個 KV head，head_dim 64）→ `q_proj` 是 576→576，`v_proj` 是 576→**192**。

## 用在哪裡
- [[make_lora_config()]]
