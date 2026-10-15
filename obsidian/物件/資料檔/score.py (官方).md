---
tags: [COALA, 物件, 資料檔, 評估]
類別: 資料檔（外部腳本）
位置: /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/score.py
---
# score.py (官方)

← [[物件 總覽]]

**是什麼**：fbai-speech 的官方評分腳本，算 WER、U-WER（常見詞）、B-WER（biasing 詞）。**一定要用官方的，數字才能和論文比。**

```bash
python /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/score.py \
    --refs /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/test-clean.biasing_500.tsv \
    --hyps exp/hyp/test-clean.s2.nobias.tsv
```

| 輸入 | 物件 |
|---|---|
| `--refs` | [[官方 ref TSV]] |
| `--hyps` | [[hyp TSV]] |

用 [[utt_id]] 對齊兩邊；hyp 少句子會報 missing（→ [[evaluate.py]] Bug B）。

論文 Table 2 對照目標：`w/o Biasing  test-clean 23.39 (4.54 / 2.34)`，格式 `B-WER (WER / U-WER)`。
