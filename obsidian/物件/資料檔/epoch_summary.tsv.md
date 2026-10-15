---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: exp/{實驗名}/epoch_summary.tsv
---
# epoch_summary.tsv

← [[物件 總覽]]

**存什麼**：每個 epoch 一列的訓練摘要，可以直接貼進 Excel 開會用。

```
epoch	step	tr_total	tr_ctc	tr_ce	val_total	val_ctc	val_ce	lr	min
```
- `tr_*`：該 epoch 所有步的平均
- `val_*`：epoch 結束時的 [[validate()]]（dev-clean 前 50 個 batch）
- `min`：從開始到現在經過幾分鐘

寫入：[[train_stage1.main()]]（不是 resume 時先寫表頭，之後每個 epoch append 一列）

> 第一次訓練（stack=4）時還沒有這個功能 → 用 [[dump_tb.py]] 從 [[tensorboard events]] 還原成 [[epoch_summary_recovered.tsv]]。
