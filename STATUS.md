# STATUS — 進度交接

> **新對話從這裡開始。** 搭配 `paper-reproduce-mentor-agent.md`（導師規則）、
> `EXPERIMENTS.md`（實驗數據）、`obsidian/`（每支檔案的詳細筆記）、
> `COALA_pipeline.drawio`（流程圖）一起看。
>
> 最後更新：2026-10-12 23:00

---

## 論文

**COALA: Robust Contextualized Speech-augmented Language Modeling for ASR via Contrastive Regularizer and Biasing Score Estimation**
arXiv:2607.08117 · INTERSPEECH 2026 · NTNU（本實驗室）· 官方 code `github.com/Guo0911/COALA`
LaTeX 原始檔在 `paper/src/main.tex`，圖在 `paper/src/images/*.png`

**定位**：教授要我自己實作一次，理解整個架構與實作上會遇到的問題。不是追平數字。

---

## 環境（每次開工先做）

```bash
conda activate coala
cd /mnt/disk2/M11515125/COALA_reproduce
export HF_HOME=/mnt/disk2/M11515125/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # ⚠️ 不加會抓到 A5000 而非 3090
export CUDA_VISIBLE_DEVICES=1            # RTX 3090（與論文同硬體）
```

| GPU | 型號 | VRAM | 用途 |
|---|---|---|---|
| 0 | RTX A5000 | 24564 MiB | 平行實驗 / 評估 |
| **1** | **RTX 3090** | **24576 MiB** | ⭐ 與論文同款，主訓練 |
| 2, 3 | RTX 2080 Ti | 11264 MiB | ❌ 太小 |

Python 3.10.21 / torch 2.5.1+cu121 / transformers **5.17.0**（⚠️ 論文時期是 4.x）/ peft 0.21.0

---

## 進度總覽

```
A 資料層
   1. E⁺ 抽取                       ✅  2620/2620 對上官方答案
   2. biasing list 產生器             ✅  E⁻ 採樣，固定 seed 可重現
   3. Dataset / DataLoader           ✅  961.06 h

B 模型層
   4. audio adapter + CTC            ✅  含 ctc_compress
   5. Stage 1 ASR 訓練               ✅  兩輪完成（stack=4 / stack=2）
      5h WER 評估                    ⚠️  evaluate.py 有 2 個 bug 待修
   6. discriminative projector       ☐  ← 下一步
   7. 三個 loss                      ☐  ← 論文核心貢獻
   8. Stage 2 scoring 訓練           ☐  含 121 次 forward 的 KV cache

C 評估層
   9. BTI top-10 + <unbiased> 門檻   ☐
  10. Recall@X / Recall#X            ☐
  11. 接 score.py 算 WER             ⚠️  腳本有了，被 evaluate.py 的 bug 卡住
```

**已建立的檔案**（`src/`，詳細筆記見 `obsidian/`）

| 檔案 | 狀態 |
|---|---|
| `parse_trans.py` `rare_words.py` `build_manifest.py` `figure1.py` `biasing.py` `dataset.py` | ✅ |
| `modules.py` `prompt.py` `lora_setup.py` `model.py` | ✅ |
| `overfit_test.py` `train_stage1.py` | ✅ |
| `check_models.py` `check_shape.py` `dump_tb.py` | ✅ 一次性工具 |
| `test.py` | 🗒️ 實驗沙盒，建議改名 `scratch.py` |
| `evaluate.py` | ⚠️ 能跑但有 2 個 bug |

**還沒建立**：`projector.py` `losses.py` `train_stage2.py` `bti.py` `recall.py`

---

## 🔴 立即要做的四件事（上次中斷在這裡）

### 1. 還原實驗 1 的學習曲線

```bash
python src/dump_tb.py exp/stage1_tc100
```

當時 `epoch_summary.tsv` 還沒實作，但 tensorboard events 檔完整保留。
產出 `scalars_raw.tsv` + `epoch_summary_recovered.tsv`。

### 2. 修 `evaluate.py` 的兩個 bug

**Bug A — 生成文字含換行符，破壞 TSV**（hyp 檔 2629 行 vs 實際 2611 句）

```python
text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
return " ".join(text.split())      # \n \t 多重空白全部收成單一空格
```

**Bug B — 評估時誤用了訓練用的超長句過濾**（少 9 句 → `score.py` 報 missing）

```python
ds = LibriSpeechDataset(args.manifest, fe, max_duration=1e9)
```

> 訓練時過濾超長句是對的（不要教模型輸出沒聽到的字），
> **但評估時不該過濾** —— 測試集固定，少測 9 句 WER 就跟論文不可比。

### 3. 測 stack=2 的 `ctc_compress` 比值 ← **最關鍵的未完成驗證**

改 `src/test.py`：`CKPT = "exp/stage1_tc100_s2/best.pt"`、`COALAModel(stack=2)`

```bash
python src/test.py
```

**判準**：stack=4 是 **0.33**，stack=2 應明顯往 **1.0** 靠近。
這是「CTC 真的學到對齊」的直接證據，目前核心發現缺的最後一塊。

### 4. 跑兩組完整 WER（修完 bug 後，各約 2 小時，可雙卡並行）

```bash
# stack=4（舊）
CUDA_VISIBLE_DEVICES=0 nohup python src/evaluate.py \
    --ckpt exp/stage1_tc100/best.pt --stack 4 \
    --manifest data/manifest/test-clean.tsv \
    --out exp/hyp/test-clean.s4.nobias.tsv --mode nobias \
    > exp/eval_s4.log 2>&1 &

# stack=2（新）
CUDA_VISIBLE_DEVICES=1 nohup python src/evaluate.py \
    --ckpt exp/stage1_tc100_s2/best.pt --stack 2 \
    --manifest data/manifest/test-clean.tsv \
    --out exp/hyp/test-clean.s2.nobias.tsv --mode nobias \
    > exp/eval_s2.log 2>&1 &

# 跑完各自算分
python /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/score.py \
    --refs /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/test-clean.biasing_500.tsv \
    --hyps exp/hyp/test-clean.s2.nobias.tsv
```

對照論文 Table 2 的 `w/o Biasing: 23.39 (4.54 / 2.34)`（格式是 `B-WER (WER / U-WER)`）。

---

## 之後的主線：Stage 2

論文核心貢獻全在這裡。順序：

### 6. `projector.py` — discriminative projector（論文 Eq. 1、2）

```
(B, L, 576) → Linear(576, D') → ReLU → Linear(D', 1) → (B, L, 1)
            → 沿 L 取平均（長度正規化）→ 每個 entity 一個純量分數 s_i
```

* `D = 576` 已實測確認（SmolLM2 的 hidden_size）
* **`D'` 論文未給**。預算推算：5.7M − adapter/CTC 1.49M − LoRA×2 1.84M ≈ 2.4M 可用。
  若 `D' = 576` 則 projector 約 333K，很寬裕
* Eq. 1 的長度正規化：`s_i = (1/L_i) Σ_j f(h_ij)`，讓長短 entity 可公平比較

### 7. `losses.py` — 三個 loss（論文 Eq. 3、4、5-7）← **核心貢獻**

**完全不依賴模型**，輸入只是「一串分數 + 哪些是正/負樣本」，可用手寫假資料驗證。

手算驗證題（論文圖 2b 的例子）：

| i | entity | s_i |
|---|---|---|
| 0 | `<unbiased>` | −2.0 |
| 1 | Watson | 3.0 |
| 2 | solved | 1.0 |
| 3 | survey | −1.0 |
| 4 | collect | −3.0 |

E⁺ = {1, 2}、E⁻ = {0, 3, 4}。用三個公式各算一次，跟 code 的輸出對照。

### 8. `train_stage2.py` — scoring 訓練

論文 3.4：**全部舊參數凍結**，只訓練新初始化的 projector + LoRA("scoring")，
**7 epochs、batch size 1、M = 120** 個候選。

:::danger
**技術難點**：instruction 是「判斷**這一個** text 在不在音檔裡」（單數）+ M=120 + batch=1
→ **每個候選各跑一次 forward，一個 utterance 共 121 次**。

`seg1 + audio` 約 105 token 對 121 次**完全相同** → **必須用 KV cache（`past_key_values`）**，
否則訓練時間不可接受。**論文完全沒提這件事怎麼處理。**
:::

### 9~11. 評估層

* `bti.py` — top-K=10 + 用 `<unbiased>` 的分數當門檻過濾
* `recall.py` — Recall@X（達到 X% recall 需要取幾個）、Recall#X（取前 X 個的 recall）
* 接 `score.py` 算完整的 WER / U-WER / B-WER

### 📌 教授交辦：學習曲線對照實驗

論文 4.1 的 claim 是**過程性**的，只能從 loss 曲線看出來，**而論文沒有提供這張圖**。

| 實驗 | Loss | 預期 |
|---|---|---|
| A | discriminative only（**不加** aux log loss） | **崩潰** ← 論文沒跑這組，關鍵對照 |
| B | discriminative + log loss (w=0.1) | 收斂（論文 baseline） |
| C | MPD-Loss 單獨 | 收斂 |
| D | DPD-Loss 單獨 | 收斂 |

要記錄 `train/pos_score_mean`、`train/neg_score_mean` —— 逐 step 記錄可畫出論文 Figure 4
那三種行為的**動態演化**，資訊量多於論文的靜態分布圖。

---

## ⭐ 到目前為止最重要的發現

### adapter 降採樣率與 CTC 詞彙的耦合約束

論文只寫「a dedicated CTC module is incorporated **following the audio adapter**」，
**adapter 架構與降採樣率、CTC 詞彙單位，三者全部未指定**（已 grep 確認 `main.tex` 無 conv / stride / pooling / stacking / 降採樣率的任何描述）。

**CTC 的硬約束**：位置數 ≥ 目標單位數（否則合法對齊為空集合，loss = +∞）

寫成速率形式：

```
位置數／秒  ≥  CTC 單位數／秒
   ↑                  ↑
adapter 降採樣率    CTC 詞彙單位
```

> [!important] **這兩個未指定的超參數不能獨立選擇**
> 我分別做了「字元級 CTC」和「stack=4」兩個各自看起來都合理的決定，
> 組合起來卻讓 **83.2% 的訓練資料在數學上無法用 CTC 訓練**。
>
> **論文未交代這兩者中的任何一個，也沒提到它們之間的約束關係。**

實測：

```
stack=4 → 1500/4 = 375 位置 / 30 秒 = 12.5 位置/秒
字元級語速：中位數 14.6、P95 18.0、最大 23.0 字元/秒
12.5 < 14.6  →  83.2% 的句子 CTC 無解（被 zero_infinity 歸零跳過）

stack=2 → 25.0 位置/秒 > 23.0（最大語速）→ 0.00% 無解
          且參數從 2.95M 降到 1.48M（Linear 輸入維度減半）
```

**兩個自我一致的假設**（未來可做 ablation）：

| | H1（我的選擇） | H2 |
|---|---|---|
| CTC 單位 | 字元（29） | 小型 BPE（約 1000） |
| 單位數／秒 | 14.6（最大 23） | 約 4~5 |
| 可容許的 stack | **≤ 2** | ≤ 8 |
| CTC head 參數 | 16.7K | 577K |
| 對 OOV 罕見字 | **友善**（永遠拼得出 `craswellers`） | 較差（切成碎片） |

H1 的辯護：本論文主題是罕見字 biasing，字元級對 OOV 天然友善。但這是**推測**，非論文所述。

**結果**：

| 指標 | stack=4 | stack=2 | 變化 |
|---|---:|---:|---:|
| tr_ctc 收斂值 | 2.70 | **0.775** | **−71%** |
| 最佳 val_total | 2.0163 | **0.4246** | **−79%** |
| 停滯原因 | **欠擬合** | **過擬合** | 完全相反 |

**三層證據**：理論（CTC 定義）+ 統計（83.2%）+ 實驗（−79%）

⚠️ **可以宣稱的 vs 不能宣稱的**

| 可以 | 不能 |
|---|---|
| 論文未指定 adapter 架構與 CTC 詞彙（已查證） | ~~論文的 stack 不可能是 4~~（若用 BPE 則 4 可行） |
| 預算排除「投影到 LM 的 49152 詞彙」 | ~~一定是字元級~~（小型 BPE 也塞得進 5.7M） |
| 在**字元級 CTC 前提下**，stack 必須 ≤ 2 | |
| 我的 ablation 有效：同設定只改 stack，val loss 差 79% | |

---

## 其他已確認的關鍵事實

### 用論文的 777M 反推它沒寫的設定

```
777M − encoder 636.8M − LM 134.5M = 5.7M  ← 所有可訓練元件的預算
```

| 反推出的設定 | 推理 | 驗證 |
|---|---|---|
| **CTC 詞彙不是 LM 的 49152** | 需 28.3M，超預算 5 倍 | 我選字元級（29）；⚠️ 但小型 BPE（約 1000，577K）也塞得進預算，**論文未指定** |
| **LoRA rank = 16** | 30 層 × (q_proj 18,432 + v_proj 12,288) = 921,600 | 實測 `trainable params: 921,600` 完全一致 |
| ~~adapter stack~~ | **沒有推** — `stack=4` 是任意選的，而且它一個人就吃掉 2.95M（52% 預算） | 後來才發現它與 CTC 詞彙有耦合約束 |

> ⚠️ **第 5 個要問作者的問題**：CTC 的詞彙單位是字元還是 BPE？這決定了 adapter 降採樣率的上界。

### 已實測的常數

| 常數 | 值 |
|---|---|
| Whisper encoder 輸出 | `(B, 1500, 1280)`，固定 pad 30 秒 |
| **`D`（= Eq. 2 的 D）** | **576** |
| LM vocab / 層數 | 49152 / 30 |
| CTC 詞彙 | 29 |
| 可訓練參數（stack=2） | **2,413,469** |
| 訓練資料 | 281,241 句 / **961.06 h** |
| 每步耗時（bf16） | 約 0.38 s（batch 4） |

### 兩個靜默失敗陷阱（已排除）

* **rare word 判定**：要用 `common_words_5k.txt` 取**補集**，不是查 `all_rare_words.txt`
  （後者缺 OOV 詞，只有 2512/2620 對）
* **GPU 編號**：`nvidia-smi` 用 PCI 序、CUDA 預設用 `FASTEST_FIRST`
  → 必須加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`

### 已知落差（要寫進 thesis）

| 落差 | 說明 |
|---|---|
| **訓練資料** | 論文 960h，目前只跑 train-clean-100（100h）。100h 在 epoch 5 就過擬合 |
| **N=5000 biasing list** | 官方 ref 只有 N=100/500/1000/2000，論文的 N=5000 自行採樣且未公開 seed |
| **9 項超參數** | lr / optimizer / scheduler / seed / LoRA 設定 / projector D′ / adapter 架構 / CTC 詞彙 / instruction 文字 |
| **運算精度** | 論文未指定，我用 encoder bf16（加速 3.92×）。已驗證餘弦相似度 0.9995，但**尚未做 fp32 對照** |
| **transformers 版本** | 我用 5.17.0，論文時期 4.x（v4→v5 有破壞性變更） |
| **>30 秒音檔** | 訓練時過濾 38 句（0.013%） |

> ✅ **唯一優勢**：有與論文同款同容量的 24GB RTX 3090
> → Table 2「N=5000 不做 BTI 就 OOM」**可以忠實重現**，多數 reproduce 做不到。

---

## 向教授請示（尚未問）

1. 能否直接問第一作者未公開的設定？（lr / optimizer / projector 的 D′）
2. 能否取得作者的 N=5000 biasing list 或其 seed？
3. 要不要上 960h（約 2.9 天）？
4. Stage 2 的 121 次 forward 的 KV cache，論文未提，作者怎麼做的？
5. **CTC 的詞彙單位是字元還是 BPE？adapter 的降採樣率是多少？**
   這兩者有耦合約束（位置/秒 ≥ 單位/秒），論文都未指定。

---

## 導師模式規則提醒

見 `paper-reproduce-mentor-agent.md`，重點：

* **一次只給一個小任務**，等我回報再繼續
* **先問我打算怎麼做**，但若我第一次沒答或答錯就**直接給答案**再解釋（2026-09-21 修訂）
* **資料前處理 → 快速模式**（直接給完整 code）；**模型層 → 放慢討論**（我自己寫，你 review）
* 論文核心（projector / 三個 loss / BTI / KV cache 打分）**一律我自己寫**
* **不確定的程式內容要問我貼，禁止自行生成**
* 所有路徑以 `/mnt/disk2/M11515125/COALA_reproduce/` 為根，不可寫到目錄外
