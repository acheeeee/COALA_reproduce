---
tags: [COALA, 資料層, 檔案]
層: 資料層
狀態: 完成
---

# rare_words.py

← [[資料層 總覽]] · [[COALA 索引]]

被 [[build_manifest]]、[[figure1]]、[[biasing]] 引用。

---

## 功能

判定一句話裡哪些字是 rare word，即論文 Eq. 3/4/5 的 **E⁺**。

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `load_word_list(path)` | 詞表檔路徑 | `set[str]` |
| `load_common_words(path=COMMON_WORDS_FILE)` | 同上 | `set[str]` — 5000 個常見詞（**判定用**） |
| `load_rare_words(path=RARE_WORDS_FILE)` | 同上 | `set[str]` — 209291 個 rare word（**採樣 E⁻ 用**） |
| `find_rare_words(text, common_set)` | `text: str`（小寫）、`common_set: set` | `list[str]` — 去重 + 字母排序 |

```python
WORDS_DIR         = Path("/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/words")
COMMON_WORDS_FILE = WORDS_DIR / "common_words_5k.txt"
RARE_WORDS_FILE   = WORDS_DIR / "all_rare_words.txt"
```

```python
def find_rare_words(text, common_set):
    found = set()
    for word in text.lower().split():
        if word not in common_set:      # 注意是 not in common，不是 in rare
            found.add(word)
    return sorted(found)
```

---

## ⚠️ 判定方式的坑（花最久才找到的 bug）

> [!danger] 要用 `common_words_5k.txt` 的補集，不是查 `all_rare_words.txt`
> **錯誤做法**：`word in all_rare_words` → 只有 **2512/2620** 列與官方一致。
> 漏掉的全是極罕見專有名詞：`whittaws`、`craswellers`、`andella`、`thel`、`beeder`
>
> **原因**：`all_rare_words.txt` 是從 **LibriSpeech 訓練語料**的詞彙推導的。只出現在 test set、訓練時從未見過的 **OOV 詞根本不在裡面**。
>
> **正解**（官方 README 明載）：
> | 檔案 | 正確用途 |
> |---|---|
> | `common_words_5k.txt` | **判定 rare word** → 取補集（不在 5000 常見詞裡的都算，OOV 自然包含） |
> | `all_rare_words.txt` | **只用來採樣干擾詞 E⁻** 的詞池 |
>
> 修正後 → **2620/2620 完全一致** ✅（驗證見 [[biasing]]）

這個 bug 的性質值得記住：**它不報錯**。若不做官方對照驗證，整個訓練會在錯誤的標籤上跑完，而且數字看起來都「正常」。

---

## 為什麼用 `set` 不用 `list`

28 萬句 × 平均 33 字 ≈ **930 萬次**查詢。

| 容器 | 查詢方式 | 一句 20 字的最差成本 |
|---|---|---|
| `list` | 從頭逐一比對 | 20 × 209291 ≈ **419 萬次**字串比對 |
| `set` | hash 直接算出位置 | **20 次** |

與元素數量無關。這是「能跑」與「跑不完」的差別，不是微調效能。

而且 `load_word_list` 必須在**迴圈外**呼叫一次 — 會變的放迴圈裡，不變的放外面。

---

## 詞表實測

```
  5000   common_words_5k.txt
209291   all_rare_words.txt       ← 論文 3.1 說的「209.2K」就是這個
214291   all_words.count.txt      = 5000 + 209291 ✓
```

確認 rare word 定義與論文完全一致 → 之後算出的 **B-WER 可與論文直接比較**。

---

## 驗證

```bash
python src/rare_words.py
```

要看到 3 個 `PASS`：

| 輸入 | 預期 E⁺ | 意義 |
|---|---|---|
| `when i was a young man i thought paul was making too much of his call` | `[]` | 沒有 rare word → `<unbiased>` 會是唯一正樣本 |
| `the air and the earth are curiously mated and intermingled as if...` | `['intermingled', 'mated']` | 多目標句 |
| `craswellers` | `['craswellers']` | **OOV 詞，驗證補集做法** |
