---
tags: [COALA, 函式, 評估]
種類: 函式
檔案: evaluate.py
---
# evaluate.parse_args()

← [[evaluate.py]] · [[函式 總覽]]

**功能**：定義並解析評估的命令列參數。

| 輸入 | 輸出 |
|---|---|
| 無 | [[args]]（評估版） |

重點：`--ckpt`、`--out` 必填；`--mode` 三選一（`nobias` / `oracle` / `stage1`）；`--stack` 必須和訓練時一致，否則載入時 adapter 的權重 shape 不合。

```bash
python src/evaluate.py --ckpt exp/stage1_tc100_s2/best.pt --stack 2 \
    --manifest data/manifest/test-clean.tsv \
    --out exp/hyp/test-clean.s2.nobias.tsv --mode nobias
```

被 [[evaluate.main()]] 呼叫。
