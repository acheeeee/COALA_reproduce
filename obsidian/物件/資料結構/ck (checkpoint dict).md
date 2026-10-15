---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: dict
aliases: [ck]
---
# ck (checkpoint dict)

← [[物件 總覽]]

**存什麼**：存檔／讀檔用的訓練狀態。

| 項目 | 內容 |
|---|---|
| 型別 | `dict` |
| 寫入 | [[save_ckpt()]] → `torch.save` → [[checkpoint .pt]] |
| 讀出 | `torch.load(...)`：[[train_stage1.main()]]（`--resume`）、[[evaluate.main()]] |

| 鍵 | 內容 |
|---|---|
| `"model"` | **只有可訓練參數**的 state_dict（adapter、CTC head、LoRA asr） |
| `"optimizer"` | [[opt]] 的狀態（AdamW 的一階、二階動量） |
| `"scheduler"` | [[sched]] 的狀態 |
| `"step"`、`"epoch"` | 存檔時的進度 |
| `"args"` | `vars(args)`，訓練設定 |

## 只存可訓練參數
凍結的 encoder（636.8M）和 LM（134.5M）不存 → 每個檔省約 2.5 GB。讀回來時用 `load_state_dict(..., strict=False)`：
- `missing`（缺的鍵）= 凍結參數，正常
- `unexpected`（多的鍵）必須是 0，否則代表架構對不上（例如 stack 設錯）→ `assert`
