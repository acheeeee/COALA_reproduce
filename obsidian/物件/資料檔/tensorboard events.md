---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: exp/{實驗名}/events.out.tfevents.*
---
# tensorboard events

← [[物件 總覽]]

**存什麼**：[[writer]] 寫出的所有 scalar 曲線（二進位格式）。

看法：
```bash
tensorboard --logdir exp/
```

讀回來：[[load_scalars()]] → [[sc (scalar dict)]]
