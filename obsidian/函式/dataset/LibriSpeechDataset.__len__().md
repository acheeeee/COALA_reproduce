---
tags: [COALA, 函式, 資料層]
種類: 方法
檔案: dataset.py
---
# LibriSpeechDataset.__len__()

← [[LibriSpeechDataset]] · [[dataset.py]] · [[函式 總覽]]

**功能**：回傳有幾句（過濾後）。

```python
def __len__(self):
    return len(self.rows)
```

| 輸入 | 輸出 |
|---|---|
| 無 | `int` |

`len(ds)` 會呼叫它；[[DataLoader (dl)]] 用它決定一個 epoch 有幾個 batch。
