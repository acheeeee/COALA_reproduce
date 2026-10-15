---
tags: [COALA, 物件, 張量]
類別: 張量
型別: torch.Tensor
shape: (B, seq, 49152)
dtype: float32
aliases: [a, both, out.logits]
---
# logits (LM 輸出)

← [[物件 總覽]]

**存什麼**：LM 對每個位置「下一個 token 是哪個詞」的分數，49152 類 = SmolLM2 的詞彙量。

| 項目 | 內容 |
|---|---|
| shape | `(B, seq, 49152)` |
| 來源 | [[out (LM 輸出物件)]] 的 `.logits` |

## 出現在哪

| 地方 | 用途 |
|---|---|
| [[COALAModel.forward_stage1()]] | 不直接取用，HF 內部用它算 [[ce_loss]] |
| [[lora_setup.main()]] | `a`（asr）、`s`（scoring）、`both` 三組 `(1, 20, 49152)`，比較不同 LoRA 組合的輸出差異 |
| [[which_gets_grad()]] | `logits.pow(2).mean()` 當假 loss 觸發 backward |

> 和 [[ctc_logits]]（29 類）、[[s (entity 分數)]]（每個 entity 1 個數）都不同。Stage 2 **不用**這個 49152 維的輸出，而是改用 LM 的 hidden state 經過 projector。
