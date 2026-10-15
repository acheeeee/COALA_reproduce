---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: model.py
---
# COALAModel.encode_audio()

← [[COALAModel]] · [[model.py]] · [[函式 總覽]]

**功能**：音訊特徵 → adapter 輸出 + CTC logits。encoder 在 bf16、不算梯度；之後的可訓練部分回到 fp32、要算梯度。

```python
def encode_audio(self, input_features):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = self.encoder(input_features)
    x = out.last_hidden_state.float()
    h = self.adapter(x)               # ⚠️ 必須在 with 外面
    ctc_logits = self.ctc_head(h)     # ⚠️ 必須在 with 外面
    return h, ctc_logits
```

## 輸入 → 輸出

| | shape | 物件 |
|---|---|---|
| 輸入 `input_features` | `(B, 80, 3000)` | [[input_features]] |
| 輸出 `h` | `(B, 750, 576)` | [[h (adapter 輸出)]] |
| 輸出 `ctc_logits` | `(B, 750, 29)` | [[ctc_logits]] |

## 內部物件
[[out (encoder 輸出物件)]] → [[x (encoder 輸出)]] `(B, 1500, 1280)`

## 呼叫了誰
[[encoder]]、[[AudioAdapter.forward()]]、[[CTCHead.forward()]]

## 被誰呼叫
[[COALAModel.forward_stage1()]]、[[transcribe_one()]]

> [!danger] 放錯位置的後果
> | adapter / ctc_head 放在 | 後果 |
> |---|---|
> | `no_grad` 裡 | **拿不到梯度、永遠學不到**；但 LoRA 還在學，loss 照樣下降 → 極難發現 |
> | `autocast` 裡 | 可訓練層用 bf16 算，梯度精度變差 |
>
> encoder 用 bf16 的理由：凍結 → 沒有梯度精度問題；RTX 3090 tensor core 加速 3.92×；和 fp32 輸出的餘弦相似度 0.9995。
