---
tags: [COALA, 資料層, 檔案]
層: 資料層
狀態: 完成
---

# parse_trans.py

← [[資料層 總覽]] · [[COALA 索引]]

**最底層工具**，被 [[build_manifest]] 和 [[figure1]] 引用。

---

## 功能

解析 LibriSpeech 的 `*.trans.txt`。

格式是「`utt_id` + 一個空格 + 整句逐字稿」：

```
1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES...
└──── 第一個 token 是 ID ────┘└──────── 其餘全部是逐字稿 ────────┘
```

所以要用 `split(" ", 1)` **只切第一個空格**，不能全切（逐字稿內部還有空格）。

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `parse_trans_file(path)` | `Path` / `str` — 一個 trans.txt 的路徑 | `list[tuple[str, str]]` — `[(utt_id, text), ...]`，text 是**原始大寫** |

常數：`TRANS` — 單檔測試用路徑（只有 `main()` 用）

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

---

## 逐字稿的格式特徵

- **全大寫**
- **沒有標點符號**，唯一例外是所有格撇號（`QUILTER'S`、`LINNELL'S`）

這是 LibriSpeech 的標準慣例，也是為什麼各方算 WER 的數字可以互相比較。

> [!warning] 大小寫落差
> 逐字稿全大寫，但官方 `ref/*.tsv`、`score.py`、word list 全部小寫。
> 轉換固定在 [[build_manifest]] 做（`.lower()`），**不在這裡做** — 這裡保持原始資料不動。

---

## 語法筆記

`split()` 括號裡放的是**分隔符**，不是要被切的字串（要被切的是點號左邊那個物件）：

```python
text.split()        # 用「任何空白」切，連續空白視為一個，自動忽略頭尾
text.split(" ")     # 明確用「一個空格」切，"A  B" 會得到 ["A", "", "B"]
text.split(" ", 1)  # 第二個參數 maxsplit=1 → 只切一次
```
