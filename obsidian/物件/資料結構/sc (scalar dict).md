---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: dict
aliases: [sc]
---
# sc (scalar dict)

← [[物件 總覽]]

**存什麼**：從 tensorboard events 檔讀出的所有 scalar 曲線。

| 項目 | 內容 |
|---|---|
| 型別 | `dict[str, list[tuple[int, float]]]` |
| 由誰產生 | [[load_scalars()]] |
| 用在 | [[dump_tb.main()]] |

```python
{"train/total_loss": [(50, 3.21), (100, 2.98), ...],   # (step, value)
 "val/total_loss":   [(2000, 2.10), ...],
 "train/lr":         [...], ...}
```
（數值為示意）

→ 寫成 [[scalars_raw.tsv]] 與 [[epoch_summary_recovered.tsv]]
