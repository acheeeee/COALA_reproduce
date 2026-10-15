---
tags: [COALA, 函式, 工具]
種類: 內部函式
檔案: dump_tb.py
---
# epoch_of()

← [[dump_tb.py]] · [[函式 總覽]]

**功能**：step 編號 → 屬於第幾個 epoch。定義在 [[dump_tb.main()]] 裡面（閉包，讀得到外層的 `args`）。

```python
def epoch_of(step):
    return (step - 1) // args.steps_per_epoch
```

| 輸入 | 輸出 |
|---|---|
| `step`（`int`，從 1 開始） | `int`，從 0 開始 |

`- 1` 是因為訓練迴圈先 `step += 1` 才做事 → 第一步是 1。所以 step 1~7134 → epoch 0、7135~14268 → epoch 1。

用到：[[STEPS_PER_EPOCH]]（經由 `args.steps_per_epoch`）
