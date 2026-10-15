---
tags: [COALA, 函式, 資料層]
種類: 方法
檔案: dataset.py
---
# LibriSpeechDataset.__init__()

← [[LibriSpeechDataset]] · [[dataset.py]] · [[函式 總覽]]

**功能**：讀 manifest 進記憶體（只有文字資料，不讀音檔），並過濾掉超過 30 秒的句子。

```python
def __init__(self, manifest_path, feature_extractor, max_duration=MAX_DURATION):
```

## 輸入

| 參數 | 型別 | 物件 |
|---|---|---|
| `manifest_path` | `str` | [[manifest TSV]] 的路徑 |
| `feature_extractor` | `WhisperFeatureExtractor` | [[fe]] |
| `max_duration` | `float` | 預設 [[MAX_DURATION]] = 30.0 |

## 產生的屬性
`self.fe`、`self.rows`（[[row (manifest 一列)]] 版本 2）

## 內部物件

| 名稱 | 角色 |
|---|---|
| `header` | 第一行切開的欄名 |
| `idx` | `{欄名: 索引}` → 用名字取欄，不依賴欄位順序 |
| `p` | 一行切成的欄位 list |
| `dur` | `float(p[idx["duration"]])` |
| `n_skip` | 被過濾的句數 |

## 輸出
無回傳值；印一行 `載入 N 句，跳過 M 句`（dev-clean：載入 2694、跳過 9）

`rare_words` 欄在這裡 `json.loads` 成 list。

> [!warning] 評估時要傳 `max_duration=1e9`
> 否則測試集也會少 9 句 → [[evaluate.py]] Bug B。
