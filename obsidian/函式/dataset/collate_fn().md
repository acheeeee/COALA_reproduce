---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: dataset.py
---
# collate_fn()

← [[dataset.py]] · [[函式 總覽]]

**功能**：把 B 個 sample 疊成一個 batch；長度不一的 [[ctc_ids]] 補 0 對齊。

```python
def collate_fn(batch):
```

## 輸入 → 輸出

| | 型別 | 物件 |
|---|---|---|
| 輸入 `batch` | `list[dict]`，長 B | [[sample (Dataset 單筆)]] 的 list（⚠️ 參數名叫 batch，但還不是 batch dict） |
| 輸出 | `dict` | [[batch]] |

## 內部物件

| 名稱 | shape | 怎麼來 |
|---|---|---|
| `batch_list` | B 個 `(80, 3000)` | 取出每筆的 input_features |
| `input_features` | `(B, 80, 3000)` | `torch.stack(batch_list)` |
| `ctc_lens` | `(B,)` | 每筆 `len(ctc_ids)` → [[ctc_lens]] |
| `max_len` | `int` | `ctc_lens.max()` |
| `ctc_ids` | `(B, max_len)` | 先 `torch.zeros`，再逐列填入 → [[ctc_ids]] |

```python
ctc_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
for i, b in enumerate(batch):
    ctc_ids[i, : len(b["ctc_ids"])] = b["ctc_ids"]
```

## stack vs pad

| 欄位 | 處理 | 原因 |
|---|---|---|
| `input_features` | `torch.stack` | 每筆都是 (80, 3000)，等長 |
| `ctc_ids` | 補 0 到最長 | 字元數不同 |
| 字串 / list | 收成 list | 不是 tensor |

被 [[DataLoader (dl)]] 自動呼叫（`collate_fn=collate_fn`）。
