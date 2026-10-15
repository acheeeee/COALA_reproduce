---
tags: [COALA, 物件, 模型與工具, LoRA]
類別: 模型與工具
型別: peft.PeftModel（包著 SmolLM2）
aliases: [self.lm, model]
---
# lm

← [[物件 總覽]]

**是什麼**：backbone 語言模型 SmolLM2-135M-Instruct，外面包了 peft，裡面掛著兩組 LoRA。

| 項目 | 內容 |
|---|---|
| 存在 | [[COALAModel]] 的 `self.lm` |
| 原始參數 | 134.5M，**全部凍結** |
| 結構 | 30 層、hidden 576（= [[D]]）、9 個 query head、3 個 KV head（GQA）、vocab 49152 |
| LoRA | `"asr"`（Stage 1 訓練）、`"scoring"`（Stage 2 訓練），各 921,600 參數 |
| 別名 | [[lora_setup.main()]] 裡叫 `model` |

## 建立順序（不能反）
```python
self.lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
for p in self.lm.parameters(): p.requires_grad = False      # 1. 先凍結
self.lm = get_peft_model(self.lm, make_lora_config(), adapter_name="asr")   # 2. 再包 peft
self.lm.add_adapter("scoring", make_lora_config())
```
反過來的話，凍結那步會把 LoRA 也凍住 → 可訓練參數 0、loss 不降、**不報錯**。

## 用法

| 呼叫 | 在哪 | 回傳 |
|---|---|---|
| `self.lm(**padded)` | [[COALAModel.forward_stage1()]] | [[out (LM 輸出物件)]] |
| `self.lm.get_input_embeddings()` | 同上、[[transcribe_one()]] | [[embed_layer]] |
| `self.lm.generate(inputs_embeds=...)` | [[transcribe_one()]] | [[out_ids]] |
| `self.lm.set_adapter("asr")` | [[evaluate.main()]] | 切換使用哪組 LoRA |

切換 / 凍結 adapter 的工具見 [[set_active()]]、[[freeze_adapter()]]。
