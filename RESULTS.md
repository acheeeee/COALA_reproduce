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
| COALA − Bias-Loss | — | — | — | — | — | — |
| COALA − MPD-Loss | — | — | — | — | — | — |
| COALA − DPD-Loss | 14 | 78 | 96.45 | 90.45 | 98.26 | 93.94 |
| *（額外）COALA − disc 單獨* | — | — | — | — | — | — |

> Recall@95 用官方 code 的算法（全體第 95 百分位），和論文表格的數字同一種算法；照論文文字逐句平均是 24.8 / 91.8（clean / other）。模型用最後一個 epoch（`epoch6.pt`），和官方相同。

### ③ 960 h

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| COALA − Bias-Loss | — | — | — | — | — | — |
| COALA − MPD-Loss | — | — | — | — | — | — |
| COALA − DPD-Loss | — | — | — | — | — | — |

---

## Table 2：ASR  B-WER（WER / U-WER）

只重現論文 Table 2 的 COALA 兩列（其他 baseline 的數字是論文從原論文引用的，不重現）。

- **COALA**：整份 biasing list 直接放進 prompt（N=5000 在論文中 OOM）
- **+ BTI (DPD-Loss)**：DPD 模型打分 → top-10 + `<unbiased>` 門檻 → 放進 prompt
- w/o Biasing 和 Oracle 只跟 Stage 1 模型有關、不經過 Stage 2，所以「+ BTI」只列 N=500 / 1000 / 5000

每格是 **B-WER (WER / U-WER)**。同一列直接比 ① 論文、② 100 h、③ 960 h。

### test-clean

| 方法 | 條件 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| COALA | w/o Biasing | 23.39 (4.54 / 2.34) | 49.35 (11.59 / 6.94) | — |
| COALA | Oracle | **1.77** (1.90 / 1.92) | 6.09 (4.76 / 4.59) | — |
| COALA | N=500 | 20.38 (4.36 / 2.49) | — | — |
| COALA | N=1000 | 21.98 (4.56 / 2.53) | — | — |
| COALA | N=5000 | OOM | — | — |
| + BTI (DPD-Loss) | N=500 | **3.25** (2.28 / 2.17) | — | — |
| + BTI (DPD-Loss) | N=1000 | **3.86** (2.50 / 2.34) | — | — |
| + BTI (DPD-Loss) | N=5000 | **6.96** (3.39 / 2.97) | — | — |

> 100 h 的 Table 2：**fbai 定義、greedy 解碼**（`score_wer.py`），完整紀錄在伺服器的 `exp/wer_all.tsv`。
> 同一格用官方 code 的定義（Whisper 正規化、官方 rare word 判定）：
>
> | 格子 | 官方定義 | 日期 |
> |---|---|---|
> | w/o clean | 48.02 (11.56 / 7.30) | 2026-10-22 |
> | Oracle clean | 5.79 (4.63 / 4.50) | 2026-10-22 |

### test-other

| 方法 | 條件 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| COALA | w/o Biasing | 39.49 (8.75 / 5.48) | — | — |
| COALA | Oracle | **6.12** (4.86 / 4.73) | — | — |
| COALA | N=500 | 35.01 (8.87 / 6.09) | — | — |
| COALA | N=1000 | 37.54 (9.18 / 6.16) | — | — |
| COALA | N=5000 | OOM | — | — |
| + BTI (DPD-Loss) | N=500 | **9.13** (5.57 / 5.20) | — | — |
| + BTI (DPD-Loss) | N=1000 | **10.59** (5.98 / 5.49) | — | — |
| + BTI (DPD-Loss) | N=5000 | **15.17** (7.33 / 6.50) | — | — |

> N=5000 的「COALA」那格：論文寫 OOM。我們會實際跑一次（`--mode file` 放整份 5000 詞的清單）。不管有沒有 OOM，結果都記下來。

---

## 圖

| 圖 | 內容 | ① | ② 100 h | ③ 960 h |
|---|---|---|---|---|
| Fig. 3 | B-WER vs 每句 target 數（1 / 2-5 / 6-24），不同 filter 方法，DPD | 有 | — | — |
| Fig. 4 | 正負樣本分數分布（boxplot），Bias / MPD / DPD × 1 / 2-5 / 6-24 | 有 | — | — |
| **額外** | 四組 loss 的學習曲線：`train/loss`、`pos_score_mean`、`neg_score_mean`（**教授交辦，論文沒有**） | 無 | ⏳ | — |

---

## 產生每一格的指令

以 DPD、test-clean、100 h 為例。換 split 就把 `test-clean` 改成 `test-other`；換模型就改 `exp/stage2_dpd`。

| 格子 | 指令 |
|---|---|
| Table 1 | `score_stage2.py --ckpt exp/stage2_dpd/best.pt --split test-clean --n 5000 --out exp/score/dpd.test-clean.N5000.jsonl` → `recall.py exp/score/dpd.test-clean.N5000.jsonl` |
| Table 2 w/o | `evaluate.py --ckpt exp/stage1_tc100_s2/best.pt --manifest data/manifest/test-clean.tsv --mode nobias --out …` → `score.py` |
| Table 2 Oracle | 同上，`--mode oracle` |
| Table 2 COALA、N | `score_stage2.py … --n N --dump-lists exp/lists/test-clean.N{N}.json` → `evaluate.py --mode file --entities exp/lists/test-clean.N{N}.json` → `score.py` |
| Table 2 + BTI、N | `score_stage2.py … --n N` → `bti.py … --out exp/bti/dpd.test-clean.N{N}.json` → `evaluate.py --mode file --entities exp/bti/…` → `score.py` |

`score.py` 用官方 ref 算分：`--refs …/ref/test-clean.biasing_1000.tsv`。B-WER 只看 ref 的第 3 欄（句子裡的 rare word），和 N 無關，所以 N=5000 也用這個 ref。
