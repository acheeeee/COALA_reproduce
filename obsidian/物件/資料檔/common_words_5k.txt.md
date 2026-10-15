---
tags: [COALA, 物件, 資料檔, 詞表]
類別: 資料檔
位置: /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words/common_words_5k.txt
---
# common_words_5k.txt

← [[物件 總覽]]

**存什麼**：LibriSpeech 訓練語料中最常見的 5000 個詞，一行一個。

來源：fbai-speech `is21_deep_bias`（Le et al. 2021 的 biasing 設定，論文沿用）。

**用途：判定 rare word** —— 不在這 5000 個裡的字都算 rare。

路徑常數 [[COMMON_WORDS_FILE]] → [[load_common_words()]] → [[common_set]]
