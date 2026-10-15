---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: exp/{實驗名}/scalars_raw.tsv
---
# scalars_raw.tsv

← [[物件 總覽]]

**存什麼**：tensorboard 所有 scalar 的長格式傾倒，方便自己畫圖。

```
tag	step	value
train/ce_loss	50	3.214567
...
```
（數值為示意）

寫入：[[dump_tb.main()]]
