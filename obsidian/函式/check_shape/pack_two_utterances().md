---
tags: [COALA, 函式, 工具]
種類: 函式
檔案: check_shape.py
---
# pack_two_utterances()

← [[check_shape.py]] · [[函式 總覽]]

**功能**：從 manifest 挑出**最短**和**最長**的兩句。

```python
def pack_two_utterances(manifest_path):
```

| | 物件 |
|---|---|
| 輸入 | [[MANIFEST]]（[[manifest TSV]]） |
| 輸出 | `[最短, 最長]`，各是 `(utt_id, audio_path, duration)` |

## 內部物件

| 名稱 | 內容 |
|---|---|
| `header` | 表頭 |
| `i_utt`, `i_path`, `i_duration` | `header.index(...)` 找欄位位置 |
| `rows` | [[row (manifest 一列)]] 的 tuple 版（版本 3） |

```python
rows.sort(key=lambda r: r[2])     # 依 duration 排序
return [rows[0], rows[-1]]
```

## 為什麼挑最短和最長
驗證「不管音檔多長，encoder 輸出都是 `(1, 1500, 1280)`」→ 證明 Whisper 固定 pad 到 30 秒。

被 [[check_shape.main()]] 呼叫。
