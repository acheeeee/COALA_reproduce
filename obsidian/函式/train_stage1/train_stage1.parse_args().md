---
tags: [COALA, 函式, 訓練]
種類: 函式
檔案: train_stage1.py
---
# train_stage1.parse_args()

← [[train_stage1.py]] · [[函式 總覽]]

**功能**：定義並解析訓練的命令列參數。

| 輸入 | 輸出 |
|---|---|
| 無（讀 `sys.argv`） | [[args]]（`argparse.Namespace`） |

完整參數表見 [[args]]。重點：`--stack` 預設 2、`--epochs 10`、`--batch-size 4`（論文 3.4），`--lr 1e-4` 和 `--warmup-ratio 0.03` 是自選。

```bash
python src/train_stage1.py --stack 2 --out exp/stage1_tc100_s2
```

被 [[train_stage1.main()]] 呼叫。
