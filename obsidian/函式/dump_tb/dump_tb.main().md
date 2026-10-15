---
tags: [COALA, 函式, 工具]
種類: main
檔案: dump_tb.py
---
# dump_tb.main()

← [[dump_tb.py]] · [[函式 總覽]]

**功能**：第一次訓練（stack=4）時還沒有 epoch 摘要功能，從 tensorboard 紀錄還原出來。

```bash
python src/dump_tb.py exp/stage1_tc100
```

## 步驟
1. argparse：`logdir`、`--steps-per-epoch`（預設 [[STEPS_PER_EPOCH]]）
2. [[load_scalars()]] → [[sc (scalar dict)]]
3. 全部倒成長格式 → [[scalars_raw.tsv]]
4. 定義 [[epoch_of()]]，算出總 epoch 數 `n_ep`
5. 每個 epoch 一列 → [[epoch_summary_recovered.tsv]]：
   - train 三個 loss：該 epoch 內所有點的**平均**
   - val 三個 loss：該 epoch 內**最後一筆**
   - lr：最後一筆
   - 沒資料就填 `-`
6. 印出摘要內容

## 內部物件
`raw`、`out`（輸出路徑）、`tr_keys`、`val_keys`、`n_ep`、`row`（一列的字串 list）、`vals`、`lrs`
