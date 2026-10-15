---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: modules.py
---
# CTCHead.__init__()

← [[CTCHead]] · [[modules.py]] · [[函式 總覽]]

```python
def __init__(self, d_in=576, vocab_size=29):
    super().__init__()
    self.proj = nn.Linear(d_in, vocab_size)
```

| 參數 | 預設 | 意義 |
|---|---|---|
| `d_in` | 576 | [[h (adapter 輸出)]] 的最後一維 |
| `vocab_size` | 29 | blank + [[CTC_CHARS]] 的 28 個 |
