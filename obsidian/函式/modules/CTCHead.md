---
tags: [COALA, 函式, 模型層, 類別]
種類: 類別
檔案: modules.py
---
# CTCHead

← [[modules.py]] · [[函式 總覽]]

**功能**：把 adapter 的輸出投影到 29 個 CTC 類別（blank + 28 字元）。

```python
class CTCHead(nn.Module):
    def __init__(self, d_in=576, vocab_size=29):
        super().__init__()
        self.proj = nn.Linear(d_in, vocab_size)
    def forward(self, h):
        return self.proj(h)
```

## 方法
[[CTCHead.__init__()]]、[[CTCHead.forward()]]

## 屬性
`self.proj` = `nn.Linear(576, 29)` → 576 × 29 + 29 = **16,733** 參數

## 為什麼是 29 而不是 LM 的 49152
576 × 49152 = 28.3M，是整個 5.7M 可訓練預算的 5 倍。（小型 BPE ≈1000 類也塞得進預算，論文沒說是哪種 → 疑點 #6。）

## 實例
[[COALAModel]] 的 `self.ctc_head`
