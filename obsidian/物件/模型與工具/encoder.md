---
tags: [COALA, 物件, 模型與工具, 音訊]
類別: 模型與工具
型別: WhisperEncoder（nn.Module）
aliases: [self.encoder, whisper]
---
# encoder

← [[物件 總覽]]

**是什麼**：Whisper-large-v2 的 encoder 部分，把 log-Mel 頻譜編碼成聲學向量。**全程凍結**。

| 項目 | 內容 |
|---|---|
| 建立 | `WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32).get_encoder()` |
| 存在 | [[COALAModel]] 的 `self.encoder` |
| 參數量 | 636.8M（論文寫 640M） |
| 層數 / d_model | 32 / 1280 |
| 狀態 | `.eval()` + 全部 `requires_grad = False` |

`whisper`（整個 WhisperModel，含用不到的 decoder）只在 `__init__` 裡暫存，取出 encoder 後就丟掉。

## 怎麼被呼叫（[[COALAModel.encode_audio()]]）
```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = self.encoder(input_features)
```
- `no_grad`：凍結，不用存中間結果 → 省記憶體
- `autocast bf16`：RTX 3090 的 tensor core 加速，759 ms → 194 ms（**3.92×**）

## 輸入 → 輸出
[[input_features]] `(B, 80, 3000)` → [[out (encoder 輸出物件)]] → [[x (encoder 輸出)]] `(B, 1500, 1280)`

> [!warning] train() 之後要再呼叫 `model.encoder.eval()`
> `model.train()` 會把所有子模組切成訓練模式（開 dropout）。凍結的 encoder 不該開，所以 [[train_stage1.main()]]、[[validate()]]、[[overfit_test.main()]] 都會補一句 `model.encoder.eval()`。
