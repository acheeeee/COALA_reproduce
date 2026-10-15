---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: figure1.py
---
# bucket_of()

← [[figure1.py]] · [[函式 總覽]]

**功能**：把「這句有幾個 rare word」歸到論文 Figure 1 的四個桶之一。

```python
def bucket_of(n):
    if n == 0: return "0"
    if n == 1: return "1"
    if n <= 5: return "2-5"
    return "6+"
```

| | 型別 | 內容 |
|---|---|---|
| 輸入 `n` | `int` | rare word 數量 |
| 輸出 | `str` | [[BUCKET_ORDER]] 裡的其中一個 |

被 [[figure1.main()]] 呼叫。
