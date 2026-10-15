---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: parse_trans.py
---
# parse_trans_file()

← [[parse_trans.py]] · [[函式 總覽]]

**功能**：讀一個 [[trans.txt]]，拆成 `(utt_id, text)` 的清單。

```python
def parse_trans_file(path):
```

## 輸入 → 輸出

| | 名稱 | 型別 | 物件 |
|---|---|---|---|
| 輸入 | `path` | `Path` / `str` | 一個 [[trans.txt]] 的路徑 |
| 輸出 | | `list[tuple[str, str]]` | [[items (trans 解析結果)]] |

## 內部物件

| 名稱 | 型別 | 角色 |
|---|---|---|
| `items` | `list` | 累積結果 |
| `line` | `str` | 一行，先 `.strip()` 去頭尾空白；空行跳過 |
| `utt_id`, `text` | `str` | `line.split(" ", 1)` —— **只切第一個空格**，逐字稿裡的空格保留 |

## 被誰呼叫
[[build_manifest()]]、[[figure1.main()]]、[[parse_trans.main()]]

## 原始碼
```python
def parse_trans_file(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, text = line.split(" ", 1)
            items.append((utt_id, text))
    return items
```

> [!tip] `split(" ", 1)` 的 `1`
> 最多切一刀 → 永遠得到兩段。不加的話整句會被切成每個字一段，`utt_id, text = ...` 會報「太多值」。
