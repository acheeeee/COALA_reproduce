---
tags: [COALA, 函式, 工具]
種類: main
檔案: check_models.py
---
# check_models.main()

← [[check_models.py]] · [[函式 總覽]]

**功能**：載入兩個預訓練模型，量參數量和關鍵維度，算出「我要自己訓練的參數預算」。

## 步驟與產出

| 步驟 | 量到的值 |
|---|---|
| 載入 WhisperModel → `get_encoder()` | encoder **636.8M**（論文 640M）、d_model 1280、32 層 |
| 載入 SmolLM2 | **134.5M**、hidden_size **576**（= [[D]]）、vocab 49152、30 層 |
| 相加，對照論文 777M | 771.3M → **差額 5.7M** = 所有可訓練元件的預算 |

## 呼叫了誰
[[count_params()]]、[[human()]]

> [!important] 這支是整個「參數預算反推」的起點
> 5.7M 這個數字後來用來推出 LoRA rank=16、排除 49152 類的 CTC 詞彙、估算 projector 的 D′ 上限。
