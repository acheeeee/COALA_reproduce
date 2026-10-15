---
tags: [COALA, 函式, 模型層]
種類: 方法
檔案: modules.py
---
# CTCHead.forward()

← [[CTCHead]] · [[modules.py]] · [[函式 總覽]]

```python
def forward(self, h):
    return self.proj(h)
```

| | shape | 物件 |
|---|---|---|
| 輸入 `h` | `(B, L, 576)` | [[h (adapter 輸出)]] |
| 輸出 | `(B, L, 29)` | [[ctc_logits]] |

`nn.Linear` 只作用在最後一維，前面的 `(B, L)` 原樣保留。

被 [[COALAModel.encode_audio()]] 呼叫（`self.ctc_head(h)`）。
