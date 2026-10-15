---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: argparse.Namespace
---
# args

← [[物件 總覽]]

**存什麼**：命令列參數。用 `args.lr` 這種屬性方式取值（`--batch-size` 會變成 `args.batch_size`）。

## [[train_stage1.parse_args()]] 的版本

| 參數 | 預設 | 意義 |
|---|---|---|
| `--stack` | 2 | [[AudioAdapter]] 的降採樣倍率 |
| `--train` / `--val` | train-clean-100 / dev-clean 的 manifest | |
| `--out` | `exp/stage1_tc100` | 輸出目錄 |
| `--epochs` | 10 | 論文 3.4 |
| `--batch-size` | 4 | 論文 3.4 |
| `--lr` | 1e-4 | 論文未給 |
| `--warmup-ratio` | 0.03 | 前 3% 步數線性 warmup |
| `--grad-clip` | 1.0 | |
| `--log-every` / `--val-every` | 50 / 2000 | 步 |
| `--seed` | 42 | |
| `--num-workers` | 8 | DataLoader |
| `--resume` | None | 從 checkpoint 續跑 |

## [[evaluate.parse_args()]] 的版本

| 參數 | 預設 | 意義 |
|---|---|---|
| `--stack` | 2 | 必須和訓練時一致 |
| `--ckpt` | 必填 | [[checkpoint .pt]] |
| `--manifest` | dev-clean | |
| `--out` | 必填 | [[hyp TSV]] 路徑 |
| `--mode` | `nobias` | `nobias` / `oracle` / `stage1`，見 [[entity_list]] |
| `--limit` | None | 只跑前 N 句 |
| `--seed` | 42 | |

[[dump_tb.main()]] 也有自己的小版本：`logdir`、`--steps-per-epoch`。

`vars(args)` 會被存進 [[ck (checkpoint dict)]] 的 `"args"`，事後可以查這個 checkpoint 是用什麼設定訓練的。
