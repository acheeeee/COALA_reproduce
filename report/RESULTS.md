# RESULTS — 論文 vs 100 h vs 960 h

>**① 論文原始數據**、**② 我們的 100 h**、**③ 我們的 960 h**。
---

## 0. 三組設定對照

| 項目 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|
| Stage 1 訓練資料 | LibriSpeech train-960h | train-clean-100（28,539 句） | LibriSpeech train-960h|
| Stage 1 | 10 epochs、batch 4 | 同左 | 同左 |
| Stage 2 訓練資料 | train-960  | train-clean-100 | train-960 |
| Stage 2 每 epoch | **論文未寫** | 28,539 句（全部） | 70,000 句(`per-epoch 70000`)|
| Stage 2 | 7 epochs、batch 1、M=120 | 同左，grad accum 4 | 同左 |
| adapter / CTC | 未寫 | frame stacking ×2 + 字元 CTC（29） | 同左 |
| LoRA | 未寫（官方 code：r=8，7 種 Linear） | r=16，q_proj / v_proj | 同左 |
| N=5000 清單 | 未公開 | 自己抽，`Random(f"0-{utt_id}")` | 同左 |
| Stage 1 CTC 權重 | 論文 0.3；**官方用 `0.5`** | 0.3（照論文） | 0.3 |
| early stopping | 論文未寫；官方`patience 10`| `patience 10`（同官方） | 同左 |
| 選 checkpoint | 官方推論用最後一個 | Stage 1 `best.pt`（val loss，epoch 5）；Stage 2 最後一個 `epoch6.pt` | Stage 1 `best.pt`（val loss，step 546,000 = epoch 7 中段）；Stage 2 `epoch6.pt` |
| 解碼（Table 2） | 論文未寫；官方 beam 20 + sampling + repetition penalty 1.1 + no-repeat 3-gram | **greedy + repetition penalty 1.1 + no-repeat 3-gram**（`--decode greedy_norep`） | 同左 |
| GPU | RTX 3090(ASR+Scoring+評估) |RTX 3090（ASR+Scoring: DPD&disc);RTX A5000（Scoring: MPD&Bias-Loss）；RTX 2080 Ti （評估） | RTX 2080 Ti（ASR）；RTX 3090 / RTX A5000（Scoring+評估） |

> 70,000：論文只寫 7 epochs、batch size 1 
> 70,000 來自官方 code（`github.com/Guo0911/COALA`，commit `7b2e008`）

>train-clean-100    28,539 句   100.6 h
>train-clean-360  104,014 句   363.6 h
>train-other-500  148,688 句   496.9 h
>───────────────────────────────────────
>合計             281,241 句   961.06 h   ← 「960 h」是取整數的說法

---

## Table 1：Biasing scoring（N = 5000）
Recall@95 越小越好；Recall#20、Recall#50 越大越好（%）。

### ① 論文

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| CTC-Filter | 94.3 | 1459.0 | 93.26 | 83.83 | 94.45 | 85.49 |
| K-Prompt | 593.3 | 1532.3 | 86.30 | 73.88 | 88.92 | 79.05 |
| Bias-Loss | 10.0 | 19.0 | 98.72 | 95.10 | 99.29 | 97.27 |
| MPD-Loss | **9.0** | 16.9 | 98.76 | 95.65 | 99.43 | 97.55 |
| DPD-Loss | **9.0** | **13.0** | **99.09** | **96.59** | **99.60** | **98.17** |

> **補充**：CTC-Filter、K-Prompt 是論文拿來比較的 baseline（別人的方法），我們不重現，只保留論文的數字。
> - **CTC-Filter**（Yang et al., SLT 2024）：CTC 解碼的逐字稿 + 拼字相似度排名
> - **K-Prompt**（Zhang et al., BigData 2023）：第 1 次轉寫 + 模糊比對
>
> 兩者都先轉成文字再比對，第一步轉錯就會連帶出錯（error propagation），所以在 test-other 上特別差。

### ② 100 h

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| COALA − Bias-Loss | 19 | 105 | 95.47 | 88.62 | 97.54 | 92.72 |
| COALA − MPD-Loss | 13 | 63 | 96.93 | 90.76 | 98.44 | 94.40 |
| COALA − DPD-Loss | 14 | 78 | 96.45 | 90.45 | 98.26 | 93.94 |
| *（額外）COALA − disc 單獨* | 15 | 83 | 96.28 | 89.82 | 98.05 | 93.58 |

> Recall@95 用官方 code 的算法（全體第 95 百分位），和論文表格的數字同一種算法；照論文文字逐句平均是 Bias 33.0 / 97.8、MPD 17.2 / 71.6、DPD 24.8 / 91.8、disc 24.3 / 82.4（clean / other）。模型用最後一個 epoch（`epoch6.pt`），和官方相同。Recall#X 是全體合計（micro），和官方 code 相同。
>
> **disc 單獨沒有崩潰**：論文第 344 行說 Bias-Loss 必須和 log loss 合用，才能避免在多目標資料上 training collapse。我們 disc 單獨訓練的 loss 停在約 1.15 下不去（梯度衝突造成的下限），但排名能力沒有變差，兩個測試集都**勝過** Bias-Loss。兩個測試集的排序都是 MPD > DPD > disc > Bias。

### ③ 960 h

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| COALA − Bias-Loss | 11 | 24 | 98.16 | 94.57 | 99.16 | 96.91 |
| COALA − MPD-Loss | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| COALA − DPD-Loss | **9** | **16** | **98.89** | **95.81** | **99.56** | **97.56** |

> Recall@95 用官方 code 的算法（同 100 h）；照論文文字逐句平均是 Bias 12.2 / 31.1、DPD 8.2 / 22.5（clean / other）。MPD 還在訓練（2026-10-29 練完 3 / 7 個 epoch）。
>
> **和論文很接近**：DPD clean 的 Recall@95 = 9，和論文的 9.0 相同；Recall#20 / #50 和論文差不到 1 個百分點（clean 98.89 vs 99.09、99.56 vs 99.60；other 95.81 vs 96.59、97.56 vs 98.17）。DPD 在每一欄都勝過 Bias-Loss，和論文的結論相同。100 h 和論文的差距主要來自訓練資料量（尤其沒有 other 語音）。
>
> 960 h 不單獨訓練 disc：Bias-Loss（0.9·disc + 0.1·log）已經包含 disc，disc 單獨的對照只在 100 h 做一次（2026-10-24 決定）。

---

## Table 2：ASR  B-WER（WER / U-WER）

只重現論文 Table 2 的 COALA 兩列（其他 baseline 的數字是論文從原論文引用的，不重現）。

- **COALA**：整份 biasing list 直接放進 prompt
- **+ BTI (DPD-Loss)**：DPD 模型打分 → top-10 + `<unbiased>` 門檻 → 放進 prompt
- w/o Biasing 和 Oracle 只跟 Stage 1 模型有關、不經過 Stage 2，所以「+ BTI」只列 N=500 / 1000 / 5000

每格是 **B-WER (WER / U-WER)**，fbai 定義（`score_wer.py`），解碼 **greedy + 防重複**（`greedy_norep`，見上方設定表）。greedy 的結果在最後的附錄。

### test-clean

| 方法 | 條件 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| COALA | w/o Biasing | 23.39 (4.54 / 2.34) | 48.41 (9.20 / 4.38) | 34.51 (6.25 / 2.77) |
| COALA | Oracle | **1.77** (1.90 / 1.92) | 6.35 (3.87 / 3.57) | 2.12 (2.06 / 2.05) |
| COALA | N=500 | 20.38 (4.36 / 2.49) | 79.14 (159.10 / 168.95) ⚠️ | 83.20 (153.21 / 161.83) ⚠️ |
| COALA | N=1000 | 21.98 (4.56 / 2.53) | 88.80 (142.77 / 149.41) ⚠️ | 98.11 (120.42 / 123.16) ⚠️ |
| COALA | N=5000 | OOM | OOM | OOM |
| + BTI (DPD-Loss) | N=500 | **3.25** (2.28 / 2.17) | 11.51 (5.05 / 4.25) | 4.86 (2.94 / 2.70) |
| + BTI (DPD-Loss) | N=1000 | **3.86** (2.50 / 2.34) | 13.87 (5.55 / 4.52) | 6.41 (3.39 / 3.02) |
| + BTI (DPD-Loss) | N=5000 | **6.96** (3.39 / 2.97) | 19.34 (6.93 / 5.41) | 10.54 (5.01 / 4.33) |

### test-other

| 方法 | 條件 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| COALA | w/o Biasing | 39.49 (8.75 / 5.48) | 61.21 (15.79 / 10.62) | 49.46 (11.42 / 7.09) |
| COALA | Oracle | **6.12** (4.86 / 4.73) | 13.78 (9.70 / 9.24) | 6.22 (5.72 / 5.66) |
| COALA | N=500 | 35.01 (8.87 / 6.09) | 85.78 (189.73 / 201.57) ⚠️ | 86.67 (174.04 / 183.99) ⚠️ |
| COALA | N=1000 | 37.54 (9.18 / 6.16) | 91.78 (168.70 / 177.46) ⚠️ | 98.92 (123.15 / 125.91) ⚠️ |
| COALA | N=5000 | OOM | OOM | OOM |
| + BTI (DPD-Loss) | N=500 | **9.13** (5.57 / 5.20) | 22.50 (11.53 / 10.28) | 11.38 (7.08 / 6.59) |
| + BTI (DPD-Loss) | N=1000 | **10.59** (5.98 / 5.49) | 24.52 (11.86 / 10.42) | 13.63 (7.83 / 7.16) |
| + BTI (DPD-Loss) | N=5000 | **15.17** (7.33 / 6.50) | 32.62 (13.80 / 11.66) | 20.36 (10.13 / 8.97) |

> **BTI 補回 w/o 和 Oracle 之間多少差距**（B-WER，(w/o − BTI) / (w/o − Oracle)）：
>
> | | clean N=500 | clean N=5000 | other N=500 | other N=5000 |
> |---|---|---|---|---|
> | ① 論文 | 93% | 76% | 91% | 73% |
> | ② 100 h | 88% | 69% | 82% | 60% |
> | ③ 960 h | 92% | 74% | 88% | 67% |
>
> 960 h 的 Oracle 已經接近論文（clean 2.12 vs 1.77、other 6.22 vs 6.12），BTI 的比例也接近論文；差距主要在 w/o（34.51 vs 23.39），也就是 Stage 1 不給提示時本身就比論文差。
>
> 960 h 的 N=5000 整份清單沒有實際跑：OOM 只跟 prompt 長度有關（同一份清單、同一個 SmolLM2），和 100 h 相同。
>
> ⚠️ **整份清單（N=500 / 1000）WER 超過 100%**：500～1000 個詞放進 prompt，遠超過訓練時的 ≤10 個，模型會照著清單唸出不在句子裡的字，產生大量插入錯誤。greedy 解碼下還會掉進重複迴圈（test-other N=500：1009 / 2939 句，34%），WER 更高（424%，見附錄）；加上防重複後迴圈消失，WER 約減半，但仍超過 100%。所以**崩潰約一半來自解碼、一半來自 prompt 太大**。BTI 只放 ≤10 個字，和訓練時相同，沒有這個問題。
>
> **N=5000 = OOM（和論文一致）**：5000 個詞的 prompt 約 18,100 個 token（test-clean 18,151、test-other 18,056），是 SmolLM2 上下文上限 8,192 的 2.2 倍；解碼時要再配置約 11.2 GB，在 24 GB 的 RTX 3090 上 OOM。論文的 OOM 定義是「推論 VRAM 超過 24 GB」，條件相同。

**同一格用官方 code 的評分定義**（Whisper 正規化、官方 rare word 判定），解碼相同：

| split | 條件 | ② 100 h | ③ 960 h |
|---|---|---|---|
| clean | w/o Biasing | 47.01 (9.10 / 4.67) | 32.80 (6.15 / 3.03) |
| clean | Oracle | 5.99 (3.76 / 3.50) | 1.88 (1.93 / 1.93) |
| clean | COALA N=500 | 77.08 (164.59 / 174.81) | 82.58 (174.51 / 185.25) |
| clean | COALA N=1000 | 87.61 (145.94 / 152.75) | 97.94 (122.62 / 125.50) |
| clean | + BTI N=500 | 10.91 (4.99 / 4.29) | 4.51 (2.83 / 2.64) |
| clean | + BTI N=1000 | 13.23 (5.50 / 4.59) | 5.90 (3.30 / 3.00) |
| clean | + BTI N=5000 | 18.57 (6.95 / 5.60) | 9.84 (5.03 / 4.47) |
| other | w/o Biasing | 59.39 (15.62 / 10.97) | 47.23 (11.24 / 7.41) |
| other | Oracle | 13.20 (9.52 / 9.13) | 5.67 (5.52 / 5.50) |
| other | COALA N=500 | 84.06 (196.04 / 207.95) | 85.95 (197.56 / 209.43) |
| other | COALA N=1000 | 90.63 (172.96 / 181.71) | 98.84 (125.48 / 128.32) |
| other | + BTI N=500 | 21.45 (11.39 / 10.32) | 10.68 (6.94 / 6.55) |
| other | + BTI N=1000 | 23.40 (11.75 / 10.51) | 12.71 (7.73 / 7.20) |
| other | + BTI N=5000 | 31.11 (13.75 / 11.90) | 19.09 (10.15 / 9.20) |

### 附錄：greedy vs 防重複（100 h，fbai 定義）

| split | 條件 | greedy | 防重複（主表） |
|---|---|---|---|
| clean | w/o Biasing | 49.35 (11.59 / 6.94) | 48.41 (9.20 / 4.38) |
| clean | Oracle | 6.09 (4.76 / 4.59) | 6.35 (3.87 / 3.57) |
| clean | COALA N=500 | 152.07 (360.36 / 385.99) | 79.14 (159.10 / 168.95) |
| clean | COALA N=1000 | 126.52 (383.58 / 415.22) | 88.80 (142.77 / 149.41) |
| clean | + BTI N=500 | 11.04 (5.59 / 4.92) | 11.51 (5.05 / 4.25) |
| clean | + BTI N=1000 | 13.47 (6.77 / 5.95) | 13.87 (5.55 / 4.52) |
| clean | + BTI N=5000 | 18.85 (8.19 / 6.88) | 19.34 (6.93 / 5.41) |
| other | w/o Biasing | 63.64 (27.26 / 23.12) | 61.21 (15.79 / 10.62) |
| other | Oracle | 18.67 (15.15 / 14.75) | 13.78 (9.70 / 9.24) |
| other | COALA N=500 | 160.00 (424.07 / 454.14) | 85.78 (189.73 / 201.57) |
| other | COALA N=1000 | 145.63 (432.37 / 465.02) | 91.78 (168.70 / 177.46) |

> - **WER、U-WER 每一格都變好**，連不放任何字的 w/o 也是（test-other U-WER 23.12 → 10.62）：greedy 本身就會在一部分句子重複，不只是整份清單。之前「100 h 的 test-other 比論文差 4 倍」有很大一部分是 greedy 造成的。
> - **BTI 與 clean Oracle 的 B-WER 略差（+0.3～0.5）**：防重複會壓低已經說過的 token。rare word 在句子裡合理地出現第二次、或被切成多個 token 時，第二次容易被擋掉。
> - test-other 的 BTI 只用防重複版跑（greedy 版停掉了，省下 3 次解碼）。

---

## 圖

| 圖 | 內容 | ① | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| Fig. 3 | B-WER vs 每句 target 數（1 / 2-5 / 6-24），不同 filter 方法，DPD | 有 | — | — |
| Fig. 4 | 正負樣本分數分布（boxplot），Bias / MPD / DPD × 1 / 2-5 / 6-24 | 有 | — | — |
| **額外** | 四組 loss 的學習曲線：`train/loss`、`pos_score_mean`、`neg_score_mean`（**教授交辦，論文沒有**） | 無 | 資料已備（`results/curves/`） | ⏳ |

---

## 產生每一格的指令

結果檔在 `c_layer_eval` 分支的 `results/`（`collect_results.py` 從伺服器的 `exp/` 整理而來，`MANIFEST.tsv` 記每個檔的來源、行數、修改時間）。

| 格子 | 指令 |
|---|---|
| Table 1 | `score_stage2.py --ckpt exp/stage2_dpd/epoch6.pt --split test-clean --n 5000 --out exp/score/dpd.e6.test-clean.N5000.jsonl` → `recall.py … --tsv exp/recall_all.tsv` |
| Table 2 w/o、Oracle | `evaluate.py --ckpt exp/stage1_tc100_s2/best.pt --manifest data/manifest/test-clean.tsv --mode nobias（或 oracle） --decode greedy_norep --out …` → `score_wer.py … --tsv exp/wer_all.tsv` |
| Table 2 BTI | `DECODE=greedy_norep bash src/run_table2.sh test-clean bti`（打分 N=500/1000/5000 → `bti.py` → 解碼 → 算分） |
| Table 2 整份清單 | `DECODE=greedy_norep bash src/run_table2.sh test-clean full` |
| 960 h | 上面的指令加 `S1=exp/stage1_960/best.pt S2=… TAG=960h` |
| 整理紀錄 | `clean_records.py`（每項只留最新、greedy 搬到附錄）→ `collect_results.py` |

`score_wer.py` 用官方 ref 算分：`--refs …/ref/test-clean.biasing_1000.tsv`。B-WER 只看 ref 的第 3 欄（句子裡的 rare word），和 N 無關，所以 N=5000 也用這個 ref。
