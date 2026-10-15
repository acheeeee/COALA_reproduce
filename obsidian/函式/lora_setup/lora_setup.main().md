---
tags: [COALA, 函式, 模型層, LoRA]
種類: main
檔案: lora_setup.py
---
# lora_setup.main()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：驗證兩組 LoRA 的建立、切換、凍結、梯度與輸出疊加行為 —— 為 Stage 2 的「兩組都啟用、只訓練 scoring」做準備。

## 步驟
**基本（TODO 1–5）**
1. 載入 LM（fp32），印總參數 134.5M
2. `get_peft_model(..., adapter_name="asr")` → 921,600 可訓練
3. `add_adapter("scoring", ...)` → 總參數再 +921,600，可訓練仍是 921,600
4. 切 asr / scoring，各印一次
5. `torch.randn(1, 20, 576)` 當 [[inputs_embeds]] → 輸出 `(1, 20, 49152)` → **證明 inputs_embeds 路線可行**

**延伸測試**

| 段落 | 測什麼 | 用到 |
|---|---|---|
| 1 | 單組啟用時 requires_grad 落在哪 | [[set_active()]]、[[which_trainable()]] |
| 2 | 兩組同時啟用 | [[set_active()]] 傳 list |
| 2b | 兩組啟用 + 凍結 asr（Stage 2 情境） | [[freeze_adapter()]] |
| 3 | backward 後誰真的有梯度 | [[which_gets_grad()]] |
| 4 | 兩組 LoRA 的輸出是否**相加** | 比較 `a`、`s`、`both` 三組 [[logits (LM 輸出)]] 的差 |

## 第 3、4 段的準備
`lora_B` 預設初始化成 0 → LoRA 一開始完全沒作用，看不出差異。所以先用不同 seed 把兩組的 `lora_B` 填成小的隨機值（×0.02），並 `model.eval()` 關掉 dropout。

## 用到的物件
[[lm]]（區域變數名 `model`）、[[lora_config]]、[[LM_ID]]
