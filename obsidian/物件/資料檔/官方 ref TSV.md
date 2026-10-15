---
tags: [COALA, 物件, 資料檔, biasing]
類別: 資料檔
位置: /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/{split}.biasing_{N}.tsv
---
# 官方 ref TSV

← [[物件 總覽]]

**存什麼**：官方提供的測試用 biasing list，每句一列。

| 欄位 | 內容 |
|---|---|
| 1 | [[utt_id]] |
| 2 | 逐字稿 |
| 3 | E⁺（JSON 陣列） |
| 4 | 完整 biasing list（JSON 陣列） |

只有 **N = 100 / 500 / 1000 / 2000**。論文用的 N = 5000 要自己抽，且論文沒公開 seed → 必然有隨機差異。

## 兩種用途
1. [[load_official_ref()]] → [[ref (官方 ref dict)]]：驗證自己的 E⁺ 抽取（2620/2620 ✓）
2. [[score.py (官方)]] 的 `--refs`：算 B-WER 時知道哪些字是 biasing word

位置常數：[[OFFICIAL_REF_DIR]]
