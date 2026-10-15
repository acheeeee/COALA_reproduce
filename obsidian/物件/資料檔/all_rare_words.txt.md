---
tags: [COALA, 物件, 資料檔, 詞表]
類別: 資料檔
位置: /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words/all_rare_words.txt
---
# all_rare_words.txt

← [[物件 總覽]]

**存什麼**：訓練語料裡所有非常見詞，209,291 個（= 論文的 209.2K），一行一個。

`5000 + 209291 = 214291` = `all_words.count.txt` 的總數 ✓

**用途：只用來抽干擾詞 E⁻**。

> [!warning] 不能拿來判定 E⁺
> 測試集有些字根本沒出現在訓練語料 → 不在這個檔裡。用它判定只有 2512/2620 對。

路徑常數 [[RARE_WORDS_FILE]] → [[load_rare_pool()]] → [[rare_pool]]
