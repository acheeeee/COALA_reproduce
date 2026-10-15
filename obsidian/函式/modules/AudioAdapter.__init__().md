---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: modules.py
---
# AudioAdapter.__init__()

← [[AudioAdapter]] · [[modules.py]] · [[函式 總覽]]

```python
def __init__(self, d_in=1280, d_out=576, stack=2):
    super().__init__()
    self.stack = stack
    self.proj = nn.Linear(d_in * stack, d_out)
```

| 參數 | 預設 | 意義 |
|---|---|---|
| `d_in` | 1280 | Whisper encoder 輸出維度（[[x (encoder 輸出)]] 的最後一維） |
| `d_out` | 576 | LM hidden size [[D]] |
| `stack` | 2 | 幾個相鄰 frame 接成一個 |

`super().__init__()`：先讓 `nn.Module` 把自己的初始化跑完（註冊參數用的內部結構），之後 `self.proj = ...` 才會被認得是子模組。
