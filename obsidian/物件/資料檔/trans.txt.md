---
tags: [COALA, 物件, 資料檔]
類別: 資料檔
位置: data/LIBRISPEECH/LibriSpeech/{split}/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt
---
# trans.txt

← [[物件 總覽]]

**存什麼**：LibriSpeech 官方逐字稿，一個 chapter 一個檔。

```
1272-128104-0000 MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
1272-128104-0001 NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
...
```
每行：`{utt_id} {全大寫逐字稿}`，第一個空格切開。

## 誰讀它
[[parse_trans_file()]] → [[items (trans 解析結果)]]；被 [[build_manifest()]]、[[figure1.main()]] 走訪（`rglob("*.trans.txt")`）

範例路徑常數：[[TRANS]]
