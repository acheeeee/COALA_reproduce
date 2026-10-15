---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# count_trainable()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：數模型裡 `requires_grad=True` 的參數總數。

```python
def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

| | 型別 |
|---|---|
| 輸入 `model` | 任何 `nn.Module` |
| 輸出 | `int` |

`p.numel()` = 這個 tensor 有幾個數字（number of elements）。

被 [[which_trainable()]]、[[lora_setup.main()]] 呼叫。其他檔案（[[model.main()]]、[[train_stage1.main()]]）用同樣的寫法直接內嵌。
