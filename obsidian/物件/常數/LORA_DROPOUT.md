---
tags: [COALA, 物件, 常數, LoRA]
類別: 常數
型別: float
---
# LORA_DROPOUT

← [[物件 總覽]]

**存什麼**：LoRA 分支輸入端的 dropout 機率。

| 項目 | 內容 |
|---|---|
| 型別 | `float` |
| 值 | `0.05` |
| 定義於 | [[lora_setup.py]] |

只在 `model.train()` 時生效。[[lora_setup.main()]] 做梯度測試前特地呼叫 `model.eval()` 關掉它，避免隨機性干擾比較。

## 用在哪裡
- [[make_lora_config()]]
