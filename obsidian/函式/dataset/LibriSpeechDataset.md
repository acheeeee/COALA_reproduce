---
tags: [COALA, 函式, 資料層, 類別]
種類: 類別
檔案: dataset.py
---
# LibriSpeechDataset

← [[dataset.py]] · [[函式 總覽]]

**功能**：PyTorch Dataset —— 把 [[manifest TSV]] 的一列變成模型能吃的 [[sample (Dataset 單筆)]]。

```python
class LibriSpeechDataset(Dataset):
```

## 方法

| 方法 | 做什麼 |
|---|---|
| [[LibriSpeechDataset.__init__()]] | 讀 manifest、過濾超長句、存成 `self.rows` |
| [[LibriSpeechDataset.__len__()]] | 回傳句數 |
| [[LibriSpeechDataset.__getitem__()]] | 讀音檔 → log-Mel、逐字稿 → CTC id |

## 屬性

| 屬性 | 物件 |
|---|---|
| `self.fe` | [[fe]] |
| `self.rows` | [[row (manifest 一列)]] 的 list（版本 2） |

## Dataset 的規則
繼承 `torch.utils.data.Dataset`，只要實作 `__len__` 和 `__getitem__`，就能交給 [[DataLoader (dl)]]。DataLoader 會呼叫 `ds[i]` 取資料、再交給 [[collate_fn()]] 打包。

**讀音檔是在 `__getitem__` 裡才做**（lazy），不是在 `__init__` 一次讀完 → 960 小時的音訊不用全部塞進記憶體。

## 實例
`ds`、`train_ds`、`val_ds`：[[train_stage1.main()]]、[[evaluate.main()]]、[[overfit_test.main()]]、[[model.main()]]、[[dataset.main()]]
