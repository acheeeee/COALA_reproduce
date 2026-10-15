---
tags: [COALA, 函式, 工具]
種類: 函式
檔案: check_models.py
---
# human()

← [[check_models.py]] · [[函式 總覽]]

**功能**：把大數字轉成易讀格式。

```python
def human(n):
    return f"{n / 1e6:.1f}M"      # 636809600 → "636.8M"
```

| 輸入 | 輸出 |
|---|---|
| `int` / `float` | `str` |

`1e6` = 1,000,000；`:.1f` = 小數點後一位。

被 [[check_models.main()]] 呼叫。
