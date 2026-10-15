---
tags: [COALA, 函式, 訓練]
種類: 函式
檔案: train_stage1.py
---
# validate()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：在 dev-clean 前 50 個 batch 上算平均 loss（不生成文字，所以快）。

```python
@torch.no_grad()
def validate(model, val_dl, rare_pool, dev, max_batches=50):
```

| 參數 | 物件 |
|---|---|
| `model` | [[COALAModel]] 實例 |
| `val_dl` | [[DataLoader (dl)]] |
| `rare_pool` | [[rare_pool]] |
| `dev` | [[dev]] |
| `max_batches` | 50（= 200 句） |
| **輸出** | `(val_total, val_ctc, val_ce)` 三個 `float` |

## 步驟
1. `model.eval()`
2. `rng = random.Random(0)` —— **每次都重建、固定 seed** → 每次驗證抽到同一組干擾詞，數字可以互相比
3. 每個 batch：[[to_device()]] → [[make_entity_lists()]] → [[COALAModel.forward_stage1()]] → 累加三個 loss
4. `model.train()` + `model.encoder.eval()`（切回訓練模式，但凍結的 encoder 保持 eval）
5. 回傳平均

## 用到的物件
[[rng]]、[[batch]]、[[entity_lists]]、[[out (stage1 loss dict)]]

`@torch.no_grad()`：整個函式不記錄計算圖 → 省記憶體、加速。

被 [[train_stage1.main()]] 呼叫（每 2000 步 + 每個 epoch 結束）。
