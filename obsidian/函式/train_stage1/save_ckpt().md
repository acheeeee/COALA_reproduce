---
tags: [COALA, 函式, 訓練]
種類: 函式
檔案: train_stage1.py
---
# save_ckpt()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：存 checkpoint，但只存可訓練參數。

```python
def save_ckpt(path, model, opt, sched, step, epoch, args):
```

| 參數 | 物件 |
|---|---|
| `path` | 例 `exp/stage1_tc100_s2/best.pt` |
| `model` | [[COALAModel]] 實例 |
| `opt` | [[opt]] |
| `sched` | [[sched]] |
| `step`, `epoch` | `int` |
| `args` | [[args]] |
| **輸出** | 無（寫檔 → [[checkpoint .pt]]，內容是 [[ck (checkpoint dict)]]） |

## 怎麼挑出可訓練參數
```python
trainable = {k: v for k, v in model.state_dict().items()
             if k in {n for n, p in model.named_parameters() if p.requires_grad}}
```
`state_dict()` 有全部參數；`named_parameters()` 帶 `requires_grad` 資訊 → 取交集。省下約 2.5 GB/檔。

> [!note] `epoch` 參數的語意不一致
> 每個 epoch 結束存的 `epoch{N}.pt` 寫入 `epoch + 1`（下一個要跑的 epoch，續跑用），而 `best.pt` 寫入當前 `epoch`。用 `best.pt` 續跑會把那個 epoch 重跑一次。

被 [[train_stage1.main()]] 呼叫。
