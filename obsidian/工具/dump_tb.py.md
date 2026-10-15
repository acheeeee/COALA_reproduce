---
tags: [COALA, 工具, 檔案]
層: 工具
狀態: 完成
---

# dump_tb.py

← [[工具 總覽]] · [[COALA 索引]]

一次性工具。不被任何檔案 import。

<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：從 tensorboard 紀錄還原 epoch 摘要。第一次訓練（stack=4）時 [[train_stage1.py]] 還沒有寫 [[epoch_summary.tsv]] 的功能，用這支補救。

| 類型 | 內容 |
|---|---|
| import 外部 | `argparse`、`pathlib`、`tensorboard`（`EventAccumulator`） |
| import 自己的 | 無 |
| 被誰 import | 無 |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[load_scalars()]] | events 檔 → [[sc (scalar dict)]] |
| [[dump_tb.main()]] | 寫出兩個 TSV |
| [[epoch_of()]] | step → epoch（main 內的閉包） |

**定義的常數**：[[STEPS_PER_EPOCH]] = 7134

**資料流**：[[tensorboard events]] → [[load_scalars()]] → [[sc (scalar dict)]] → [[scalars_raw.tsv]] + [[epoch_summary_recovered.tsv]]
<!-- 結構:end -->

---

## 用法

```bash
python src/dump_tb.py exp/stage1_tc100
```

（STATUS 待辦 #1：還原實驗 1 的學習曲線）

---

## 還原規則

| 欄 | 取法 | 為什麼 |
|---|---|---|
| train loss | 該 epoch 內所有紀錄點的平均 | 和 epoch_summary 的「epoch 平均」概念一致 |
| val loss | 該 epoch 內最後一筆 | epoch 結束時的驗證最能代表該 epoch |
| lr | 最後一筆 | |

> [!note] 和真正的 epoch_summary 會有微小差異
> train 只有每 50 步一個紀錄點；val 是每 2000 步一次，不一定剛好落在 epoch 結尾。
