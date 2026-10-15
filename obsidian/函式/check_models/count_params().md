---
tags: [COALA, 函式, 工具]
種類: 函式
檔案: check_models.py
---
# count_params()

← [[check_models.py]] · [[函式 總覽]]

**功能**：數一個模組的**全部**參數（不管可不可訓練）。

```python
def count_params(module):
    return sum(p.numel() for p in module.parameters())
```

| 輸入 | 輸出 |
|---|---|
| `nn.Module` | `int` |

和 [[count_trainable()]] 的差別：那個只數 `requires_grad=True` 的。

被 [[check_models.main()]] 呼叫。
