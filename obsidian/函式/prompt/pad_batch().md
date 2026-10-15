---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# pad_batch()

← [[prompt.py]] · [[函式 總覽]]

**功能**：把長度不同的多句三件組補齊成一個 batch。

```python
def pad_batch(samples, pad_side="right"):
```

## 輸入 → 輸出

| | 物件 |
|---|---|
| 輸入 `samples` | [[samples]]（[[sample (stage1 序列)]] 的 list） |
| 輸入 `pad_side` | `"right"`（訓練）/ `"left"`（生成） |
| 輸出 | [[padded]] |

## 內部物件

| 名稱 | 內容 |
|---|---|
| `B`、`lens`、`max_len` | 句數、每句長度、最長 |
| `d`、`dev`、`dtype` | 從第一句抄來（576、裝置、型別） |
| `embeds` | `zeros(B, max_len, d)` → batch 版 [[inputs_embeds]] |
| `labels` | `full((B, max_len), -100)` → batch 版 [[labels]] |
| `mask` | `zeros(B, max_len)` → batch 版 [[attention_mask]] |
| `sl` | `slice(0, n)`（右 pad）或 `slice(max_len-n, max_len)`（左 pad） |

```python
for i, s in enumerate(samples):
    n = lens[i]
    sl = slice(0, n) if pad_side == "right" else slice(max_len - n, max_len)
    embeds[i, sl] = s["inputs_embeds"]
    labels[i, sl] = s["labels"]
    mask[i, sl]   = s["attention_mask"]
```

## 三種 pad 值

| 欄位 | pad 值 | 為什麼 |
|---|---|---|
| embeds | 0 | 反正被 mask 擋掉 |
| labels | -100 | 不算 loss（[[IGNORE_INDEX]]） |
| mask | 0 | 告訴 attention 不要看 |

被 [[COALAModel.forward_stage1()]] 呼叫（用預設右 pad）。
