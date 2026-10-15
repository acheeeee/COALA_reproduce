---
tags: [COALA, 函式, 模型層, LoRA]
種類: 函式
檔案: lora_setup.py
---
# which_trainable()

← [[lora_setup.py]] · [[函式 總覽]]

**功能**：診斷用 —— 印出可訓練參數分別落在 asr、scoring、其他哪一組。

```python
def which_trainable(model, tag):
```

| 參數 | 意義 |
|---|---|
| `model` | [[lm]] |
| `tag` | 印出時的標籤文字 |

## 內部物件

| 名稱 | 內容 |
|---|---|
| `names` | 所有 `requires_grad=True` 的參數名 |
| `n_asr`、`n_sco` | 名字含 `.asr.` / `.scoring.` 的**張量個數** |
| `tot` | [[count_trainable()]] 的結果 |

印出：`[tag] 可訓練 921,600 | asr 張量 120 | scoring 張量 0 | 其他 0` 這種格式，以及 `model.active_adapters`。

被 [[lora_setup.main()]] 呼叫。
