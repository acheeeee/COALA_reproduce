---
tags: [COALA, 物件, 模型與工具]
類別: 模型與工具
型別: list[torch.nn.Parameter]
---
# params

← [[物件 總覽]]

**是什麼**：所有要訓練的參數（`requires_grad=True` 的那些）。

```python
params = [p for p in model.parameters() if p.requires_grad]
```

| 項目 | 內容 |
|---|---|
| 張量數 | 124（adapter 2 + CTC head 2 + LoRA 30 層 × 2 模組 × A/B 2） |
| 參數總數（stack=2） | **2,413,469** = adapter 1,475,136 + CTC head 16,733 + LoRA asr 921,600 |
| 參數總數（stack=4） | 3,888,029 |
| 定義於 | [[train_stage1.main()]]、[[overfit_test.main()]] |

## 去處
- `torch.optim.AdamW(params, ...)` → [[opt]]
- `clip_grad_norm_(params, 1.0)`

只把可訓練的交給 optimizer，凍結的 7 億多個參數完全不佔 optimizer 記憶體。
