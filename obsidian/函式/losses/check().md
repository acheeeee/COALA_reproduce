---
tags: [COALA, 函式, 模型層, loss]
種類: 函式
檔案: losses.py
---
# check()

← [[losses.py]] · [[函式 總覽]]

**功能**：測試小工具 —— 比對算出來的 loss 和手算答案，差距 < 1e-5 就印 ✅。

```python
def check(name, got, want):
    ok = abs(got.item() - want) < 1e-5
    print(f"{name:<20} {got.item():.6f}  (預期 {want:.6f})  {'✅' if ok else '❌'}")
```

| 參數 | 型別 |
|---|---|
| `name` | `str`，印出的標籤 |
| `got` | 0 維 tensor（`.item()` 轉成 Python float） |
| `want` | `float`，手算答案 |

容差 1e-5：浮點運算順序不同會有極小誤差（例如 `1 − 0.9 = 0.09999999999999998`），不能用 `==`。

被 [[losses.main()]] 呼叫。
