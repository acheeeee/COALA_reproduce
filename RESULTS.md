# RESULTS — 論文 vs 100 h vs 960 h

> 和論文 Table 1、Table 2 完全相同的格式，三組並排：**① 論文原始數據**、**② 我們的 100 h**、**③ 我們的 960 h**。
> 每跑完一項就填一格。產生每一格的指令見最下面。
>
> 圖例：— 尚未跑 · ⏳ 跑中 · ✅ 已填 · ✗ 不重現（其他論文的方法）
>
> 建立：2026-10-21

---

## 0. 三組設定對照

| 項目 | ① 論文 | ② 100 h | ③ 960 h |
|---|---|---|---|
| Stage 1 訓練資料 | LibriSpeech 960 h = train-clean-100 + train-clean-360 + train-other-500 | train-clean-100（28,539 句，100.59 h） | **和論文同一份資料**：三個 split 合併成 `train-960.tsv`（281,241 句，961.06 h）；訓練時濾掉 >30 秒的 38 句 |
| Stage 1 | 10 epochs、batch 4 | 同左 | 同左 |
| Stage 2 訓練資料 | LibriSpeech 960 h（同上） | train-clean-100 | `train-960.tsv`（同上） |
| Stage 2 每 epoch | **論文未寫**；官方 code `stage_2.sh` 第 29 行 `--train-batch-per-epoch 70000`（見下方註） | 28,539 句（全部） | 70,000 句（`--utts-per-epoch 70000`） |
| Stage 2 | 7 epochs、batch 1、M=120 | 同左，grad accum 4 | 同左 |
| adapter / CTC | 未寫 | frame stacking ×2 + 字元 CTC（29） | 同左 |
| LoRA | 未寫（官方 code：r=8，7 種 Linear） | r=16，q_proj / v_proj | 同左 |
| N=5000 清單 | 自行抽樣，未公開 | 自己抽，`Random(f"0-{utt_id}")` | 同左 |
| Stage 1 CTC 權重 | 論文 0.3（第 268 行）；**官方 `stage_1.sh` 用 `--ctc-weight 0.5`** | 0.3（照論文） | 0.3 |
| early stopping | 論文未寫；官方：看 `valid/loss`、patience **10**、min_delta 0（`train_ASR.py` 第 142–147 行、`train_Bias.py` 第 143–148 行）→ Stage 1 共 10 epochs、Stage 2 共 7 epochs，**實際上不會觸發** | `--early-stop-patience 10`（同官方） | 同左 |
| 選 checkpoint | 官方：Stage 1 看 `valid/wer`、Stage 2 看 `valid/loss`，存前 3 名 | Stage 1 看 val loss、Stage 2 看 val Recall#20，存最好的 1 個 | 同左 |
| GPU 型號（四組 Scoring 互相獨立，同一張卡只是排隊依序跑） | RTX 3090 ×1：訓練（ASR、Scoring）和評估都用這一張（第 260 行；Table 2 註「inference VRAM exceeded 24 GB」） | RTX 3090（ASR）；RTX 3090（Scoring：DPD，跑完再跑 disc）；RTX A5000（Scoring：MPD，跑完再跑 Bias-Loss）；RTX 2080 Ti ×2（評估） | RTX 2080 Ti（ASR）；RTX 3090 / RTX A5000（Scoring，預定）；RTX 2080 Ti（評估） |

> **70,000 的出處（2026-10-21 查證）**：論文只寫「7 epochs、batch size 1」（`main.tex` 第 270 行），**沒有寫每個 epoch 用幾句**。
> 70,000 來自官方 code（`github.com/Guo0911/COALA`，commit `7b2e008`）：
> - `stage_2.sh` 第 27 行 `--total-training-step 490000`、第 29 行 `--train-batch-per-epoch 70000`、第 30 行 `--grad-accumulate-steps 4`
> - `recipes/models/bias.py` 第 86 行 `max_epochs = total_training_step // train_batch_per_epoch` → 490000 // 70000 = **7**
> - `recipes/train_Bias.py` 第 168 行 `limit_train_batches=70000`，搭配第 121 行 `shuffle=True` → 每個 epoch 隨機取 7 萬句
> - 官方另外掛了 early stopping callback，可能不到 7 個 epoch 就停 → 我們沒有實作，記為差異

---

## Table 1：Biasing scoring 品質（N = 5000）

Recall@95 越小越好；Recall#20、Recall#50 越大越好（%）。

### ① 論文

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| CTC-Filter | 94.3 | 1459.0 | 93.26 | 83.83 | 94.45 | 85.49 |
| K-Prompt | 593.3 | 1532.3 | 86.30 | 73.88 | 88.92 | 79.05 |
| COALA − Bias-Loss | 10.0 | 19.0 | 98.72 | 95.10 | 99.29 | 97.27 |
| COALA − MPD-Loss | **9.0** | 16.9 | 98.76 | 95.65 | 99.43 | 97.55 |
| COALA − DPD-Loss | **9.0** | **13.0** | **99.09** | **96.59** | **99.60** | **98.17** |

### ② 100 h

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| CTC-Filter | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| K-Prompt | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| COALA − Bias-Loss | — | — | — | — | — | — |
| COALA − MPD-Loss | — | — | — | — | — | — |
| COALA − DPD-Loss | — | — | — | — | — | — |
| *（額外）COALA − disc 單獨* | — | — | — | — | — | — |

### ③ 960 h

| Methods | Recall@95 clean | Recall@95 other | Recall#20 clean | Recall#20 other | Recall#50 clean | Recall#50 other |
|---|---|---|---|---|---|---|
| CTC-Filter | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| K-Prompt | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| COALA − Bias-Loss | — | — | — | — | — | — |
| COALA − MPD-Loss | — | — | — | — | — | — |
| COALA − DPD-Loss | — | — | — | — | — | — |

---

## Table 2：ASR 結果 B-WER（WER / U-WER）

只重現論文 Table 2 的 COALA 兩列（其他 baseline 的數字是論文從原論文引用的，不重現）。

- **COALA**：整份 biasing list 直接放進 prompt（N=5000 在論文中 OOM）
- **+ BTI (DPD-Loss)**：DPD 模型打分 → top-10 + `<unbiased>` 門檻 → 放進 prompt
- w/o Biasing 和 Oracle 兩欄只跟 Stage 1 模型有關，兩列相同

### ① 論文

| Methods | w/o clean | w/o other | Oracle clean | Oracle other | N=500 clean | N=500 other | N=1000 clean | N=1000 other | N=5000 clean | N=5000 other |
|---|---|---|---|---|---|---|---|---|---|---|
| COALA | 23.39 (4.54 / 2.34) | 39.49 (8.75 / 5.48) | **1.77** (1.90 / 1.92) | **6.12** (4.86 / 4.73) | 20.38 (4.36 / 2.49) | 35.01 (8.87 / 6.09) | 21.98 (4.56 / 2.53) | 37.54 (9.18 / 6.16) | OOM | OOM |
| + BTI (DPD-Loss) | 23.39 (4.54 / 2.34) | 39.49 (8.75 / 5.48) | **1.77** (1.90 / 1.92) | **6.12** (4.86 / 4.73) | **3.25** (2.28 / 2.17) | **9.13** (5.57 / 5.20) | **3.86** (2.50 / 2.34) | **10.59** (5.98 / 5.49) | **6.96** (3.39 / 2.97) | **15.17** (7.33 / 6.50) |

### ② 100 h

| Methods | w/o clean | w/o other | Oracle clean | Oracle other | N=500 clean | N=500 other | N=1000 clean | N=1000 other | N=5000 clean | N=5000 other |
|---|---|---|---|---|---|---|---|---|---|---|
| COALA | 49.35 (11.59 / 6.94) | — | 6.09 (4.76 / 4.59) | — | — | — | — | — | — | — |
| + BTI (DPD-Loss) | （同上） | （同上） | （同上） | （同上） | — | — | — | — | — | — |

> 100 h 的 Table 2：**fbai 定義、greedy 解碼**（`score_wer.py`），完整紀錄在伺服器的 `exp/wer_all.tsv`。
> 同一格用官方 code 的定義（Whisper 正規化、官方 rare word 判定）：
>
> | 格子 | 官方定義 | 日期 |
> |---|---|---|
> | w/o clean | 48.02 (11.56 / 7.30) | 2026-10-22 |
> | Oracle clean | 5.79 (4.63 / 4.50) | 2026-10-22 |

### ③ 960 h

| Methods | w/o clean | w/o other | Oracle clean | Oracle other | N=500 clean | N=500 other | N=1000 clean | N=1000 other | N=5000 clean | N=5000 other |
|---|---|---|---|---|---|---|---|---|---|---|
| COALA | — | — | — | — | — | — | — | — | — | — |
| + BTI (DPD-Loss) | （同上） | （同上） | （同上） | （同上） | — | — | — | — | — | — |

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
