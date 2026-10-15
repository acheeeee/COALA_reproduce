---
tags: [COALA, 函式, 訓練]
種類: 函式
檔案: train_stage1.py
---
# to_device()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：把 batch 裡的三個 tensor 欄位搬上 GPU。

```python
def to_device(batch, dev):
    for k in ["input_features", "ctc_ids", "ctc_lens"]:
        batch[k] = batch[k].to(dev, non_blocking=True)
    return batch
```

| 參數 | 物件 |
|---|---|
| `batch` | [[batch]] |
| `dev` | [[dev]] |
| **輸出** | 同一個 [[batch]]（就地修改後回傳） |

`non_blocking=True` 搭配 DataLoader 的 `pin_memory=True`：搬移和 GPU 計算可以重疊，不用乾等。

字串、list 欄位（`utt_id`、`text`、`rare_words`）留在 CPU。

被 [[train_stage1.main()]]、[[validate()]] 呼叫。
