---
tags: [COALA, 函式, 資料層]
種類: main
檔案: rare_words.py
---
# rare_words.main()

← [[rare_words.py]] · [[函式 總覽]]

**功能**：用 3 句官方答案已知的例子測試 [[find_rare_words()]]，印 PASS / FAIL。

## 步驟
1. [[load_common_words()]] → [[common_set]]
2. `tests` = 3 組 `(utt_id, text, 官方答案)`：
   - 無 rare word 的句子 → `[]`
   - `"...curiously mated and intermingled..."` → `["intermingled", "mated"]`
   - OOV 字 `"craswellers"` → `["craswellers"]`（不在 [[all_rare_words.txt]] 裡，專門測補集邏輯）
3. 逐一比對、印結果

第三個測試是關鍵：用 rare 詞表判定的錯誤做法會在這裡 FAIL。
