---
tags: [COALA, 函式, 資料層, biasing]
種類: 函式
檔案: biasing.py
---
# add_unbiased()

← [[biasing.py]] · [[函式 總覽]]

**功能**：把 `<unbiased>` 放到清單最前面，成為 e₀。

```python
def add_unbiased(biasing_list):
    return [UNBIASED] + list(biasing_list)
```

| | 物件 |
|---|---|
| 輸入 | [[biasing_list]] |
| 輸出 | [[full_list (E)]]（長度 +1） |

用到：[[UNBIASED]]

`list(biasing_list)` 複製一份 → 不會改到傳進來的原清單。

被 [[biasing.main()]] 呼叫；Stage 2 會用。
