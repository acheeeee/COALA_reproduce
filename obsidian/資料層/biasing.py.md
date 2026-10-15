---
tags: [COALA, 資料層, 檔案]
層: 資料層
狀態: 完成
---

# biasing.py

← [[資料層 總覽]] · [[COALA 索引]]

引用 [[rare_words.py]]。被 [[train_stage1.py]]、[[evaluate.py]]、[[model.py]] 的 `main()` 使用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：產生完整的 biasing list E = {`<unbiased>`} ∪ E⁺ ∪ E⁻，切出 E⁺ / E⁻ 的索引，並讀官方參考答案驗證。

| 類型 | 內容 |
|---|---|
| import 外部 | `collections`、`json`、`random`、`pathlib` |
| import 自己的 | [[rare_words.py]]（`RARE_WORDS_FILE`、`load_common_words`、`load_rare_words`、`find_rare_words`） |
| 被誰 import | [[train_stage1.py]]、[[evaluate.py]]、[[overfit_test.py]]、[[model.py]]（main） |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[load_rare_pool()]] | 詞池 → 排序 list（可重現抽樣） |
| [[make_biasing_list()]] | E⁺ + n 個干擾詞 → [[biasing_list]] |
| [[make_stage1_list()]] | Stage 1 的 10 個 entity（最多 5 正） |
| [[add_unbiased()]] | 加 e₀ → [[full_list (E)]] |
| [[split_pos_neg()]] | → [[pos_idx]]、[[neg_idx]] |
| [[load_official_ref()]] | 讀 [[官方 ref TSV]] |
| [[biasing.main()]] | 三段驗證（2620/2620） |

**定義的常數**：[[UNBIASED]]、[[OFFICIAL_REF_DIR]]

**資料流**：
```
positives (E⁺) + rare_pool + rng → make_biasing_list → biasing_list
  → add_unbiased → full_list (E) → split_pos_neg → pos_idx / neg_idx
```
圖中物件：[[positives (E⁺)]]、[[rare_pool]]、[[rng]]、[[biasing_list]]、[[full_list (E)]]、[[pos_idx]]、[[neg_idx]]
<!-- 結構:end -->

---

## 功能

產生完整 biasing list（**E⁺ ∪ E⁻ ∪ {`<unbiased>`}**），並讀取官方參考檔。

[[rare_words.py]] 只給你 E⁺，但論文的三個 loss 需要完整清單 E — 其中 E⁻ 是從 209291 個 rare word 裡隨機抽出、**不在這句話裡**的干擾詞。

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `load_rare_pool(path=RARE_WORDS_FILE)` | 詞表路徑 | `list[str]` — **排序過**的 209291 個詞 |
| `make_biasing_list(positives, rare_pool, n_distractors, rng)` | `list[str]`、`list`、`int`、`random.Random` | `list[str]` — 正樣本 + n 個干擾詞，字母排序，**不含** `<unbiased>` |
| `make_stage1_list(positives, rare_pool, rng, max_pos=5, n_total=10)` | 同上 | `list[str]` — 長度 10，最多 5 個正樣本 |
| `add_unbiased(biasing_list)` | `list[str]` | `list[str]` — `<unbiased>` 插最前面（= e₀） |
| `split_pos_neg(full_list, positives)` | `list[str]`、`list[str]` | `(pos_idx: list[int], neg_idx: list[int])` |
| `load_official_ref(split, n)` | `str` 例 `"test-clean"`、`int` | `dict[utt_id] -> {"text", "positives", "biasing_list"}` |

常數：`UNBIASED = "<unbiased>"`、`OFFICIAL_REF_DIR`

---

## `<unbiased>` 的角色（論文 2.1）

> [!important] 它的身分是**條件性**的
> - 句中**有** rare word → E⁺ = 那些 rare word，`<unbiased>` 屬於 **E⁻**
> - 句中**沒有** rare word → E⁺ = `{<unbiased>}`，其餘全是 E⁻
>
> 這個分支就是 `split_pos_neg` 在做的事。之後三個 loss 全部依賴它給出正確的 E⁺ / E⁻。

它的顆粒度是**整句層級**（「這整句裡 list 的候選一個都沒出現」），不是掛在某個字上的標記。

**第二個用途（論文 3.4）**：推論時用它的分數當**門檻**，分數比它低的候選直接丟掉。

```python
def split_pos_neg(full_list, positives):
    pos_set = set(positives)
    if pos_set:
        pos_idx = [i for i, e in enumerate(full_list) if e in pos_set]
    else:
        pos_idx = [full_list.index(UNBIASED)]
    pos_idx_set = set(pos_idx)
    neg_idx = [i for i in range(len(full_list)) if i not in pos_idx_set]
    return pos_idx, neg_idx
```

---

## 三個設計要點

> [!tip] 1. `rng` 從外部傳入，不在函式裡建
> `random.Random(42)` 建立一個**獨立**的產生器，不影響全域 `random`。當參數傳進去就能精確控制「這次採樣用哪個 seed」，測試可重複。
>
> 論文**未公開 seed**（TODO 疑點 #2），所以自己的 seed 必須固定並記錄。

> [!tip] 2. `load_rare_pool` 回傳 `sorted(...)` 而非 set
> `rng.sample()` 需要**有序容器**。對 set 採樣的結果會依 Python 內部 hash 順序而變，不同執行可能不同 → **破壞可重現性**。這是很容易忽略的細節。

> [!tip] 3. `make_biasing_list` 的多抽策略
> 要 `n_distractors` 個負樣本，但隨機抽可能抽到正樣本。**多抽 `len(pos)` 個**，最壞情況下全部正樣本都被抽中，濾掉後仍剩足夠。數學上保證一次抽足，不用 while 迴圈重試。
>
> ```python
> sampled = rng.sample(rare_pool, n_distractors + len(pos))
> negatives = [w for w in sampled if w not in pos_set][:n_distractors]
> return sorted(pos + negatives)
> ```

---

## `make_stage1_list` 的 `max_pos=5`

論文 3.4：
> *"prompts are constructed by sampling up to 5 positive entities within a 10-entity set"*

即使句子有 24 個 rare word 也只放 5 個。目的是 **prevent over-reliance on biasing information** — 不讓模型養成「答案一定都在提示裡」的習慣。

---

## 用官方 2620 列資料做的驗證

```bash
python src/biasing.py
```

```
官方 test-clean.biasing_500.tsv 共 2620 列
  E+ 是 E 的子集合          : 2620/2620     ← 官方檔本身的性質，驗證 TSV 解析正確
  我的 find_rare_words 一致 : 2620/2620     ← ⭐ 驗證 E⁺ 抽取邏輯與作者完全一致
```

**第二項是這支檔案最重要的產出。** 它用 2620 列真實資料驗證 [[rare_words.py]] 的實作 —— 遠比手動比對兩句可靠。

> [!danger] 這個驗證抓到了真正的 bug
> 第一次跑只有 **2512/2620**。印出不一致的例子後發現漏掉 `whittaws`、`craswellers` 等 OOV 詞 → 追出「判定方式用錯詞表」（詳見 [[rare_words.py]]）。
>
> 修正後 2620/2620。**若跳過這個驗證，整個訓練會在錯誤標籤上跑完而不報錯。**

### 自我驗證的 pattern

```python
tests = [(id, 輸入, 預期輸出), ...]
for utt_id, text, expected in tests:
    got = find_rare_words(text, rare_set)
    mark = "PASS" if got == expected else "FAIL"
```

把「輸入」和「已知正確答案」成對存起來，用迴圈比對 — 單元測試的雛形。

**reproduce 的鐵則**：每個中間結果只要能找到權威答案（官方 ref 檔、論文表格數字），就寫成這種對照，**不要靠肉眼看輸出「感覺對」**。否則等訓練跑完才發現指標算錯，幾十小時就白費了。

---

## 官方 list 長度不固定（已釐清，可忽略）

`len(list) − len(positives)` 的分布：

```
500 -> 1713 列 (65.4%)
499 ->  672 列 (25.6%)
498 ->  195 列
497 ->   34 列
496 ->    5 列
495 ->    1 列        平均缺 0.45
```

這個形狀（指數衰減、平均不到 1）是**有放回抽樣後去重**的特徵。驗算：從 209291 個詞有放回抽 500 次，期望重複數 ≈ 500²/(2×209291) ≈ **0.6**，與實測同量級。

**本檔用無放回抽樣**，每次精準給 n 個不重複干擾詞。差 0.09%，對任何指標無可測量影響。

### 對策：test set 直接用官方 ref 檔

| 用途 | 大小 | 來源 |
|---|---|---|
| Stage 1 訓練 prompt | 10 | 自產（每 epoch 重抽） |
| Stage 2 訓練候選 | M=120 | 自產 |
| N=5000 評估 | 5000 | 自產（官方無 → limitation） |
| **N=500 / 1000 評估** | — | **官方 ref 檔**（逐句對齊、零隨機性） |

訓練用的清單本來就該每次重抽（資料增強的一部分），所以有放回/無放回完全不重要。

> [!warning] N=5000 的已知落差
> 官方 ref 只提供 N=100/500/1000/2000，**沒有 N=5000**。但論文 Table 1（Recall 全表）和 Table 2 的 N=5000 欄都用這個規模 → 作者自行隨機採樣且**未公開 seed**。
>
> → N=500/1000 可逐句對齊；**N=5000 的數字必然有隨機差異，不應視為重現失敗**。這是明確的 limitation。

---

## `load_official_ref` 讀的四欄格式

| 欄 | 內容 | 對應論文符號 |
|---|---|---|
| 1 | utterance ID | — |
| 2 | 參考文字（小寫） | Y |
| 3 | 句中的 rare words | **E⁺**（可能是 `[]`） |
| 4 | 完整 biasing list（含 distractors） | **E**（第 3 欄的超集） |

第 3、4 欄是 **JSON 陣列**（用 `json.loads()` 解析），且**照字母排序**（不是出現順序 → 正樣本位置不會洩漏資訊給模型）。

⚠️ `<unbiased>` **不在官方檔裡** — 它是 COALA 自己加的 e₀，所以要用 `add_unbiased()` 補上。
