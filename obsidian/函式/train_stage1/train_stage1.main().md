---
tags: [COALA, 函式, 訓練]
種類: main
檔案: train_stage1.py
---
# train_stage1.main()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：Stage 1 完整訓練迴圈。

## 1. 準備

| 步驟 | 物件 |
|---|---|
| [[train_stage1.parse_args()]]、`torch.manual_seed` | [[args]] |
| 輸出目錄、[[writer]]、[[epoch_summary.tsv]] 表頭（非續跑時） | |
| [[fe]] → `train_ds`、`val_ds`（[[LibriSpeechDataset]]）→ `train_dl`、`val_dl` | [[DataLoader (dl)]] |
| [[load_rare_pool()]]、`Random(args.seed)` | [[rare_pool]]、[[rng]] |
| `COALAModel(stack=args.stack)`，`model.train()` + `model.encoder.eval()` | [[COALAModel]] |
| 可訓練參數 → AdamW → linear warmup 排程 | [[params]]、[[opt]]、[[sched]] |
| `--resume`：載入 [[ck (checkpoint dict)]]，`strict=False`，assert 沒有 unexpected 鍵 | |

## 2. 每一步
```python
batch = to_device(batch, dev)
el = make_entity_lists(batch["rare_words"], rare_pool, rng)
opt.zero_grad()
out = model.forward_stage1(batch, el)
out["loss"].backward()
torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
opt.step()
sched.step()
```
- 累加 `ep_tot / ep_ctc / ep_ce / ep_n` 算 epoch 平均
- 每 50 步：寫 tensorboard、更新 tqdm 進度條
- 每 2000 步：[[validate()]]，創新低就存 `best.pt`

## 3. 每個 epoch 結束
[[validate()]] → 寫一列 [[epoch_summary.tsv]] → 寫 `epoch/*` tag → 檢查 best → 存 `epoch{N}.pt`

## 呼叫了誰
[[train_stage1.parse_args()]]、[[LibriSpeechDataset]]、[[collate_fn()]]、[[load_rare_pool()]]、[[COALAModel]]、[[to_device()]]、[[make_entity_lists()]]、[[COALAModel.forward_stage1()]]、[[validate()]]、[[save_ckpt()]]

## 內部的純量物件

| 名稱 | 角色 |
|---|---|
| `total_steps` | `len(train_dl) × epochs` |
| `start_epoch`, `step` | 進度（續跑時從 checkpoint 還原） |
| `best_val` | 目前最低的 val_total |
| `ep_tot`, `ep_ctc`, `ep_ce`, `ep_n` | epoch 內累加器 |
| `t_start`, `mins` | 計時 |
| `line` | 寫進 epoch_summary 的一列 |

> `clip_grad_norm_`：梯度總長度超過 1.0 就等比例縮小，避免單步更新太大。
