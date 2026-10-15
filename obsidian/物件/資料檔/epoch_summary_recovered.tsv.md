---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: exp/{實驗名}/epoch_summary_recovered.tsv
---
# epoch_summary_recovered.tsv

← [[物件 總覽]]

**存什麼**：從 tensorboard 還原的 epoch 摘要，格式和 [[epoch_summary.tsv]] 幾乎一樣（少了 `min` 欄）。

| 欄 | 怎麼算 |
|---|---|
| `tr_*` | 該 epoch 內所有 train 紀錄點的**平均** |
| `val_*` | 該 epoch 內**最後一筆** val 紀錄 |
| `lr` | 該 epoch 最後一筆 |

> [!note] 和真正的 epoch_summary 有微小差異
> train 平均只用每 50 步一個的紀錄點（不是每一步），val 是 epoch 內最後一次的每 2000 步驗證，不一定剛好在 epoch 結尾。

寫入：[[dump_tb.main()]]
