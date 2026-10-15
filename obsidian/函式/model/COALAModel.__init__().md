---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: model.py
---
# COALAModel.__init__()

← [[COALAModel]] · [[model.py]] · [[函式 總覽]]

**功能**：載入兩個預訓練模型、建立新模組、掛 LoRA、設定凍結。

```python
def __init__(self, stack=2):
```

| 參數 | 預設 | 意義 |
|---|---|---|
| `stack` | 2 | 傳給 [[AudioAdapter]]；evaluate 時必須和訓練時相同 |

## 四個步驟
```python
# 1. Whisper encoder：只要 encoder，凍結 + eval
whisper = WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32)
self.encoder = whisper.get_encoder().eval()
for p in self.encoder.parameters(): p.requires_grad = False

# 2. 新模組（可訓練）
self.adapter = AudioAdapter(stack=stack)
self.ctc_head = CTCHead()

# 3. LM：先凍結，再包 peft（順序不能反）
self.lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
for n in self.lm.parameters(): n.requires_grad = False
self.lm = get_peft_model(self.lm, make_lora_config(), adapter_name="asr")
self.lm.add_adapter("scoring", make_lora_config())

# 4. tokenizer
self.tokenizer = AutoTokenizer.from_pretrained(LM_ID)
```

## 產生的屬性
[[encoder]]、`self.adapter`（[[AudioAdapter]]）、`self.ctc_head`（[[CTCHead]]）、[[lm]]、[[tokenizer]]

## 呼叫了誰
[[AudioAdapter.__init__()]]、[[CTCHead.__init__()]]、[[make_lora_config()]]

## 用到的常數
[[WHISPER_ID]]、[[LM_ID]]

> [!warning] 一律 `dtype=torch.float32`
> transformers 5.x 會沿用 checkpoint 的 dtype（SmolLM2 是 bf16、Whisper 是 fp32）→ 不指定會在接起來時 dtype 不合。
> 第 3 步迴圈變數取名 `n` 但其實是參數 `p`，不影響功能。
