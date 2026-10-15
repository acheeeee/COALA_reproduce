---
tags: [COALA, 函式, 模型層, CTC]
種類: 函式
檔案: modules.py
---
# ctc_compress()

← [[modules.py]] · [[函式 總覽]]

**功能**：用 CTC 的預測挑出「有意義的位置」—— 去掉 blank 和連續重複，只留這些位置的 576 維向量給 LM。

```python
def ctc_compress(hidden, logits, blank_id=0):
```

## 輸入 → 輸出

| | 名稱 | shape | 物件 |
|---|---|---|---|
| 輸入 | `hidden` | `(L, 576)`，**單句** | [[h (adapter 輸出)]]`[i]` |
| 輸入 | `logits` | `(L, 29)`，**單句** | [[ctc_logits]]`[i]` |
| 輸入 | `blank_id` | `int`，預設 0 | [[CTC_BLANK_ID]] |
| 輸出 | | `(L', 576)` | [[audio_tok]] |

## 內部物件（依序產生）

| 物件 | shape | 計算 |
|---|---|---|
| [[pred]] | `(L,)` | `logits.argmax(dim=-1)` |
| [[not_blank]] | `(L,)` | `pred != blank_id` |
| [[change]] | `(L-1,)` | `pred[1:] != pred[:-1]` |
| `first` | `(1,)` | `torch.ones(1, dtype=bool, device=pred.device)` |
| [[changed]] | `(L,)` | `torch.cat([first, change])` |
| [[keep]] | `(L,)` | `not_blank & changed` |

```python
return hidden[keep]
```

## 被誰呼叫
[[COALAModel.forward_stage1()]]（迴圈內逐句）、[[transcribe_one()]]

## 重點
- **只用 logits 做決定，回傳的是 hidden** —— 餵給 LM 的是 576 維聲學向量，不是字元
- `argmax` 不可微分，但沒關係：梯度只需要流過「被選中的 hidden 列」，[[model.main()]] 實測 124/124 張量都有梯度
- 一次只能處理一句（每句 L' 不同）

> [!warning] 輸出長度 L' 直接反映 CTC 學得好不好
> stack=4 實測 L' 只有字元數的 0.33 → LM 只拿到 1/3 的音訊資訊，轉寫從句中開始飄。stack=2 的比值尚未測（STATUS 待辦 #3）。
