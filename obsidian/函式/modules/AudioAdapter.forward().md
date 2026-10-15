---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: modules.py
---
# AudioAdapter.forward()

← [[AudioAdapter]] · [[modules.py]] · [[函式 總覽]]

**功能**：frame stacking（相鄰 `stack` 個向量接成一個長向量）+ Linear 投影。

```python
def forward(self, x):
    B, T, d_in = x.shape
    x = x.reshape(B, T // self.stack, d_in * self.stack)
    return self.proj(x)
```

## shape 變化（stack=2）
```
x   (B, 1500, 1280)                       ← x (encoder 輸出)
 ↓ reshape：每 2 個相鄰 frame 首尾相接
    (B,  750, 2560)
 ↓ self.proj = Linear(2560, 576)
    (B,  750,  576)                       → h (adapter 輸出)
```
圖中物件：[[x (encoder 輸出)]]、[[h (adapter 輸出)]]

| | 物件 |
|---|---|
| 輸入 `x` | [[x (encoder 輸出)]] |
| 輸出 | [[h (adapter 輸出)]] |

## 為什麼 reshape 就等於「把相鄰 frame 接起來」
tensor 在記憶體裡是照最後一維連續排的。`(1500, 1280)` 看成 `(750, 2560)` 時，第 0 列 = 原本第 0 列的 1280 個數 + 第 1 列的 1280 個數。**不需要搬動資料，也沒有資訊損失**。

被 [[COALAModel.encode_audio()]] 呼叫（`self.adapter(x)`）。

> `T // self.stack` 是整數除法：1500 能被 2、4 整除，所以剛好。若 T 不能整除，reshape 會報錯。
