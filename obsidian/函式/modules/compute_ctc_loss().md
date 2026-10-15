---
tags: [COALA, 函式, 模型層, CTC, loss]
種類: 函式
檔案: modules.py
---
# compute_ctc_loss()

← [[modules.py]] · [[函式 總覽]]

**功能**：包裝 `F.ctc_loss`，處理好維度順序、log 機率、輸入長度。

```python
def compute_ctc_loss(logits, targets, target_lengths, blank_id=0):
```

## 輸入 → 輸出

| 參數 | shape | 物件 |
|---|---|---|
| `logits` | `(B, T, C)` | [[ctc_logits]] |
| `targets` | `(B, S)` | [[ctc_ids]] |
| `target_lengths` | `(B,)` | [[ctc_lens]] |
| `blank_id` | `int` | [[CTC_BLANK_ID]] |
| **輸出** | `()` | [[ctc_loss]] |

## 內部物件

| 物件 | shape | 計算 |
|---|---|---|
| [[log_probs]] | `(T, B, C)` | `logits.log_softmax(dim=-1).transpose(0, 1)` |
| `B`, `T` | `int` | `logits.shape[0]`, `logits.shape[1]` |
| [[input_lengths]] | `(B,)` | `torch.full((B,), T, ...)` |

```python
return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                  blank=blank_id, reduction="mean", zero_infinity=True)
```

## 三個要注意的地方
1. `F.ctc_loss` 要 **(T, B, C)** → 要 `transpose(0, 1)`
2. 要吃 **log 機率** → 先 `log_softmax`
3. `input_lengths` 全部是 T（Whisper 固定 30 秒，輸入端沒有 padding）

## `zero_infinity=True` 的雙面性
無解的句子 loss = inf → 歸零，訓練不會爆掉。**但也讓問題被藏起來**：stack=4 時 83% 的句子都被靜默歸零，只表現為 loss 卡在 2.7。

## 被誰呼叫
[[COALAModel.forward_stage1()]]、[[modules.main()]]

量級：隨機初始化約 10（[[modules.main()]] 實測 10.18）。
