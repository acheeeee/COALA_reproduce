---
tags: [COALA, 索引]
---

# COALA 索引

論文：**COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation**
arXiv:2607.08117 · INTERSPEECH 2026 · NTNU · 官方 code: `github.com/Guo0911/COALA`

流程圖：`../docs/COALA_pipeline.drawio`（拖進 https://app.diagrams.net/ 開啟）
完整紀錄：`../docs/TODO.md`

---

## 🧭 三層閱讀法（2026-10-16 新增）

筆記分三層，**由小到大**：

| 層 | 單位 | 每篇記什麼 | 入口 | 篇數 |
|---|---|---|---|---|
| 1 | **物件**（常數、tensor、資料結構、模型零件、資料檔） | 存什麼、shape / dtype / 型別、從哪來到哪去 | [[物件 總覽]] | 134 |
| 2 | **函式 / 類別 / 方法** | 功能、輸入、輸出、內部用到哪些物件、呼叫誰、被誰呼叫、原始碼 | [[函式 總覽]] | 83 |
| 3 | **檔案（.py）** | 每篇開頭的「🧩 結構」：import 誰、被誰 import、定義了哪些函式、整合出什麼功能、資料流 | 下方表格 | 19 |

**怎麼讀**：從檔案筆記的「🧩 結構」看全貌 → 點函式看細節 → 點物件看 shape。反過來，從物件筆記的「從哪來 → 到哪去」可以追一個 tensor 的一生（例：[[input_features]] → [[x (encoder 輸出)]] → [[h (adapter 輸出)]] → [[audio_tok]] → [[inputs_embeds]]）。

> Obsidian 的 **Graph view** 和每篇底部的 **Backlinks** 會顯示「誰連到這篇」，就是「這個物件被哪些函式用到」。

---

## 一個檔案一個筆記

### [[資料層 總覽]]

| 檔案 | 做什麼 | 狀態 |
|---|---|---|
| [[parse_trans.py]] | 解析 `trans.txt` → `[(utt_id, text)]` | ✅ |
| [[rare_words.py]] | 判定 rare word → **E⁺**（用 common 取補集） | ✅ |
| [[build_manifest.py]] | 一次性建表 → `data/manifest/*.tsv` | ✅ |
| [[figure1.py]] | 重現論文 Figure 1 的分布 | ✅ |
| [[biasing.py]] | **E⁻** 採樣、`<unbiased>`、讀官方 ref | ✅ |
| [[dataset.py]] | manifest → log-Mel tensor | ✅ |

### [[模型層 總覽]]

| 檔案 | 做什麼 | 狀態 |
|---|---|---|
| [[modules.py]] | AudioAdapter / CTCHead / ctc_compress / CTC loss | ✅ |
| [[prompt.py]] | embedding 查表 + 四段拼接 + labels 遮罩 | ✅ |
| [[lora_setup.py]] | LoRA r=16，兩組具名 adapter | ✅ |
| [[model.py]] | **COALAModel** 組裝者 | ✅ |
| [[projector.py]] | discriminative projector（Eq. 1、2）→ 每個 entity 一個分數 | ✅ |
| [[losses.py]] | disc / MPD / DPD / Bias-Loss（Eq. 3–7）← **核心貢獻** | ✅ |

### [[訓練與評估 總覽]]

| 檔案 | 做什麼 | 狀態 |
|---|---|---|
| [[overfit_test.py]] | 2 句 × 200 步，架構驗證 | ✅ |
| [[train_stage1.py]] | Stage 1 ASR 訓練 | 🔄 |
| [[evaluate.py]] | `generate()` → 接 score.py | ☐ |

### [[工具 總覽]]

| 檔案 | 產出的關鍵數字 | 狀態 |
|---|---|---|
| [[check_models.py]] | `D=576`、**預算 5.7M** | ✅ |
| [[check_shape.py]] | `(B,1500,1280)`、固定 pad 30 秒 | ✅ |
| [[dump_tb.py]] | 從 tensorboard 還原 epoch 摘要 | ✅ |
| [[test.py]] | 實驗沙盒（無保存價值） | 🗒️ |
| [[效能分析與 bf16]] | encoder 佔 82%、**bf16 快 3.92x**（主題筆記） | ✅ |

---

## 環境

```
伺服器：bedo@79
工作目錄：/mnt/disk2/M11515125/COALA_reproduce     ← 所有東西只能放這裡
外部唯讀：/mnt/disk2/M11515125/fbai-speech         （word list / score.py / ref）
         /mnt/disk2/LIBRISPEECH                   （symlink 到 data/LIBRISPEECH）
HF 快取：/mnt/disk2/M11515125/hf_cache
```

每次開工：

```bash
conda activate coala
cd /mnt/disk2/M11515125/COALA_reproduce
export HF_HOME=/mnt/disk2/M11515125/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # ⚠️ 不加會抓到 A5000 而非 3090
export CUDA_VISIBLE_DEVICES=1            # RTX 3090（與論文同硬體）
```

| GPU | 型號 | VRAM | 用途 |
|---|---|---|---|
| 0 | RTX A5000 | 24564 MiB | 開發 / 平行實驗 |
| **1** | **RTX 3090** | **24576 MiB** | ⭐ 與論文同款同容量 |
| 2, 3 | RTX 2080 Ti | 11264 MiB | ❌ 太小 |

Python 3.10.21 / torch 2.5.1+cu121 / **transformers 5.17.0**（⚠️ 論文時期是 4.x）/ peft 0.21.0

---

## 關鍵常數（全部實測）

| 常數 | 值 |
|---|---|
| Whisper encoder 輸出 | `(B, 1500, 1280)` |
| **`D`（LM hidden size）** | **576** ← Eq. 2 的 `D` |
| LM vocab / 層數 | 49152 / 30 |
| CTC 詞彙 | **29**（空格 + 撇號 + a-z + blank） |
| Audio token（adapter 後） | stack=2：**750**（stack=4：375）→ `ctc_compress` → 理想 ≈ 字元數（例 89） |
| **可訓練預算** | **5.7M** = 777M − 636.8M − 134.5M |
| 實際可訓練（Stage 1） | stack=2：**2,413,469**（stack=4：3,888,029） |
| projector（D′=576） | **332,929** |
| LoRA rank | **16**（由預算反推，實測吻合） |
| 訓練資料 | 281241 句 / **961.06 h** |

---

## 進度

```
A 資料層  1 E⁺ 抽取 ✅   2 biasing list ✅   3 Dataset ✅
B 模型層  4 adapter+CTC ✅   5 Stage 1 訓練 ✅（兩輪：stack=4 / stack=2）
             5a prompt ✅  5b LoRA ✅  5c CTC loss ✅
             5d 組裝 ✅    5e 過擬合 ✅  5f 腳本 ✅  5g 訓練 ✅
             5h WER 評估 ⚠️ evaluate.py 有 2 個 bug 待修
          6 projector ✅   7 三個 loss ✅   8 Stage 2：8a prompt ✅ / 8b forward + KV cache ☐ ← 下一步
C 評估層  9 BTI ☐   10 Recall ☐   11 WER ⚠️
```

**→ 詳細的交接狀態、待辦、下一步見 `../docs/STATUS.md`**

---

## ⭐ 目前最重要的發現：adapter 降採樣率讓 CTC 在數學上失效

論文未給 adapter 架構。CTC 的硬約束是**位置數 ≥ 字元數**：

```
stack=4 → 12.5 位置/秒  vs  實測語速中位數 14.6 字元/秒  →  83.2% 句子無解
stack=2 → 25.0 位置/秒  vs  最大語速 23.0 字元/秒        →   0.00% 無解
```

| 指標 | stack=4 | stack=2 | 變化 |
|---|---:|---:|---:|
| tr_ctc 收斂值 | 2.70 | **0.775** | **−71%** |
| 最佳 val_total | 2.0163 | **0.4246** | **−79%** |
| 停滯原因 | **欠擬合** | **過擬合** | 完全相反 |

完整數據見 `../docs/EXPERIMENTS.md`

---

---

## 論文的三個核心 claim（要驗證的目標）

1. **DPD / MPD-Loss 可單獨收斂**，不需 baseline Bias-Loss 所需的 auxiliary log loss（w=0.1）
2. **Biasing scoring 品質**：N=5000 時 DPD-Loss 在 test-clean `Recall#20 = 99.09%`、test-other `96.59%`
3. **接上 BTI（top-K=10）後 B-WER 大幅下降**：N=5000 從 **OOM** → 6.96 (clean) / 15.17 (other)

> [!important] 📌 教授交辦：學習曲線
> claim 1 是**過程性**的，只能從 **loss 曲線**看出來（Table 1 的 Recall 看不出來），
> **而論文沒有提供這張圖** → 這會是我自己產出、論文沒有的證據。
>
> 四組對照實驗見 [[訓練與評估 總覽]]。

---

## 已知落差（要寫進 thesis）

| 落差 | 說明 |
|---|---|
| **N=5000 的 biasing list** | 官方 ref 只有 N=100/500/1000/2000，論文的 N=5000 是自行採樣且未公開 seed → 必然有隨機差異 |
| **訓練資料規模** | 論文 960h，先用 train-clean-100（100h）打通 |
| **9 項超參數** | lr / optimizer / scheduler / seed / LoRA 設定 / projector D′ / adapter 架構 / CTC 詞彙 / instruction 文字，全部論文未給 |
| **transformers 版本** | 我用 5.17.0，論文時期是 4.x（v4→v5 有破壞性變更） |
| **>30 秒音檔** | 38 句（0.013%）被過濾 |

> [!success] 唯一的優勢
> 伺服器上有**與論文同款同容量的 24GB RTX 3090** → Table 2「N=5000 不做 BTI 就 OOM」這格**可以忠實重現**。多數 reproduce 因硬體不同無法驗證此項。
