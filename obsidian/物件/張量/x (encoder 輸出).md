---
tags: [COALA, 物件, 張量, 音訊]
類別: 張量
型別: torch.Tensor
shape: (B, 1500, 1280)
dtype: float32（encoder 內部 bf16）
aliases: [x, hid, last_hidden_state]
---
# x (encoder 輸出)

← [[物件 總覽]]

**存什麼**：Whisper encoder 對 30 秒音訊的編碼，每 20 ms 一個 1280 維向量。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor` |
| shape | `(B, 1500, 1280)` |
| dtype | encoder 在 bf16 autocast 下輸出 bf16 → 立刻 `.float()` 轉回 **float32** |
| 別名 | `out.last_hidden_state`、`hid`（[[check_shape.py]]）、[[AudioAdapter.forward()]] 的參數 `x` |

## 維度意義

| 維度 | 大小 | 意義 |
|---|---|---|
| 1500 | 時間步 | 3000 frames 經 encoder 的 conv（stride 2）減半 → 每步 20 ms |
| 1280 | d_model | Whisper-large-v2 的 hidden size |

## 從哪來 → 到哪去
```
input_features (B,80,3000) ──encoder (凍結, no_grad, bf16)──→ x (B,1500,1280)
                               ──AudioAdapter──→ h (adapter 輸出)
```
圖中物件：[[input_features]]、[[encoder]]、[[AudioAdapter]]、[[h (adapter 輸出)]]
- 產生：[[COALAModel.encode_audio()]]（來自 [[out (encoder 輸出物件)]]）
- 使用：[[AudioAdapter.forward()]]

> [!warning] `.float()` 之後的計算必須在 `no_grad` / `autocast` 區塊**外面**
> 否則 adapter 拿不到梯度，或用 bf16 算梯度。見 [[COALAModel.encode_audio()]]。
