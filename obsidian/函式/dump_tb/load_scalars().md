---
tags: [COALA, 函式, 工具]
種類: 函式
檔案: dump_tb.py
---
# load_scalars()

← [[dump_tb.py]] · [[函式 總覽]]

**功能**：讀一個 tensorboard 紀錄資料夾裡的所有 scalar。

```python
def load_scalars(logdir):
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}
```

| | 物件 |
|---|---|
| 輸入 `logdir` | 例 `exp/stage1_tc100`（含 [[tensorboard events]]） |
| 輸出 | [[sc (scalar dict)]] |

## 內部物件

| 名稱 | 角色 |
|---|---|
| `ea` | `EventAccumulator`，tensorboard 官方的讀取器 |
| `tags` | 所有 scalar 名稱 |

> `size_guidance={"scalars": 0}` = **全部讀進來**。預設只保留一部分取樣點（為了省記憶體），會漏資料。

被 [[dump_tb.main()]] 呼叫。
