# 實驗紀錄 EXPERIMENTS

> 每跑完一個實驗就往下累加。開會時直接開這個檔案。

---

## ⭐ 核心發現：論文未交代的 adapter 降採樣率會讓 CTC 在數學上失效

### 問題

論文 3.4 只寫「a dedicated CTC module is incorporated **following the audio adapter**」，
**完全沒給 audio adapter 的架構或降採樣率**（TODO 疑點 #5）。

我依「frame stacking k=4」實作，結果 CTC loss 在 epoch 6 後卡在 **2.7** 不再下降。

### 診斷

CTC 的硬約束是「**時間步數 ≥ 字元數**」（重複字元還要額外插 blank）。

```
Whisper encoder 1500 frames 涵蓋固定 30 秒
  → AudioAdapter(stack=4) → 375 個位置
  → 375 ÷ 30 = 每秒只有 12.5 個位置
```

實測 LibriSpeech train-clean-100 的語速：

| 統計量 | 字元/秒 |
|---|---|
| 中位數 | **14.6** |
| P95 | 18.0 |
| 最大 | 23.0 |

**12.5 位置/秒 < 14.6 字元/秒 → 位置不夠，CTC 無解。**

### 量化規模

```bash
awk -F'\t' 'NR>1 {nc=length($5); if (nc > $3*POS_PER_SEC) c++; t++} \
    END {printf "%d/%d (%.2f%%) 無解\n", c, t, c/t*100}' \
    data/manifest/train-clean-100.tsv data/manifest/dev-clean.tsv
```

| 設定 | 位置/秒 | CTC 無解的句子 |
|---|---|---|
| **stack=4** | 12.5 | **25997 / 31243 = 83.2%** ❌ |
| **stack=2** | 25.0 | **1 / 31243 = 0.00%** ✅ |

→ 那 83.2% 的句子 loss 是 `inf`，被 `zero_infinity=True` 歸零跳過，
**等於 83% 的訓練資料對 CTC 完全沒有訓練效果**。這就是 loss 卡在 2.7 的原因。

### 連鎖後果（一條因果鏈貫穿到最終轉寫品質）

```
CTC 沒學好
  → 預測出的非 blank 位置太少（只有實際字元數的 33%，實測見下）
  → ctc_compress 只保留那些位置 → 84% 的語音資訊被丟棄
  → LM 只拿到 27 個向量卻要重建 89 個字元的句子
  → 前段有音訊依據、後段只能靠語言模型先驗猜測
  → 轉寫從句中開始飄移
```

### 反推論文的設定

既然論文的 CTC 能運作，它的 adapter 降採樣率**必須**滿足「位置/秒 > 語速」，
所以 **stack 不可能是 4**。這是從資料反推出論文未交代設定的另一個例子
（前例：用 777M 總參數量反推 CTC 詞彙大小與 LoRA rank）。

### 修法與附帶好處

| | stack=4 | stack=2 |
|---|---|---|
| adapter 位置數 | 375 | 750 |
| 每秒位置數 | 12.5 ❌ | **25** ✅ |
| `Linear` 維度 | `5120 → 576` | `2560 → 576` |
| **adapter 參數量** | 2,949,696 | **1,475,136**（少一半） |
| 可訓練總參數 | 3,888,029 | **2,413,469** |

參數變少是因為輸入維度減半。而且**不增加 LM 負擔** —— LM 看到的是 `ctc_compress`
**之後**的數量，取決於內容（約 89 個），不是壓縮前的 375 或 750。

---

## 實驗 1：Stage 1 ASR，stack=4（基準 / 失敗案例）

**設定**

| 項目 | 值 | 來源 |
|---|---|---|
| 訓練資料 | train-clean-100（28539 句 / 100.59 h） | ⚠️ downscale，論文用 960h |
| 驗證資料 | dev-clean（2694 句，跳過 9 句 >30s） | |
| epochs / batch | 10 / 4 | 論文 3.4 |
| CTC / CE 權重 | 0.3 / 1.0 | 論文 3.4 |
| optimizer | AdamW, lr=1e-4, wd=0.01 | **論文未給** |
| scheduler | linear warmup 3% + linear decay | **論文未給** |
| grad clip / seed | 1.0 / 42 | **論文未給** |
| adapter | frame stacking **stack=4** | **論文未給** |
| LoRA | r=16, alpha=32, dropout=0.05, q_proj+v_proj | **論文未給**（由參數預算反推） |
| 可訓練參數 | 3,888,029 / 5.7M 預算 | |
| 硬體 | RTX 3090 24GB（與論文同款同容量） | |
| 耗時 | 8.5 小時（51 分/epoch，2.32 it/s） | |

**輸出**：`exp/stage1_tc100/`（best.pt、epoch0~9.pt、tensorboard events）

### Loss 曲線

訓練 loss（每 epoch 最後一個 batch）：

| epoch | ce | ctc | total |
|---|---|---|---|
| 初始 | 5.73 | 12.80 | 9.57 |
| 7 | 0.582 | 2.72 | 1.40 |
| 8 | 0.315 | 2.66 | 1.11 |
| 9 | 0.441 | 2.70 | 1.25 |

驗證 loss（`grep "New best"`）：

```
Step  2000 → 4.7307
Step  4000 → 3.3502
Step  6000 → 2.9078
Step 10000 → 2.6214
Step 18000 → 2.3470
Step 24000 → 2.3281
Step 26000 → 2.1253
Step 30000 → 2.0935
Step 46000 → 2.0163   ← best.pt（epoch 6）
（step 46000 之後到 71340 沒有再進步）
```

**兩個觀察：**

1. **CE 下降一個數量級**（5.73 → 0.3~0.6），模型確實學到轉寫
2. **CTC 從 epoch 7 到 9 幾乎不動**（2.72 → 2.66 → 2.70）→ 上述 83% 無解問題
3. **有效訓練長度只有約 6.4 epochs**。論文說 10 epochs，但在 100h 資料上 step 46000 之後 val loss 不再改善（後 3.5 epoch / 約 3 小時無貢獻）。val loss 單調下降無反彈 → 不是過擬合，是資料量或容量限制

### ctc_compress 對齊品質檢查

判準：壓縮後 token 數 / 實際字元數 → 接近 1.0 代表 CTC 學到對齊。

| utt_id | 壓縮後 | 實際字元 | 比值 |
|---|---|---|---|
| 1272-128104-0000 | 27 | 89 | 0.30 |
| 1272-128104-0001 | 17 | 63 | 0.27 |
| 1272-128104-0002 | 80 | 172 | 0.47 |
| 1272-128104-0003 | 47 | 133 | 0.35 |
| 1272-128104-0004 | 132 | 362 | 0.36 |
| 1272-128104-0005 | 30 | 108 | 0.28 |
| 1272-128104-0006 | 24 | 74 | 0.32 |
| 1272-128104-0007 | 27 | 99 | 0.27 |
| **平均** | | | **0.33** |

（訓練前隨機初始化時是 158/89 = 1.78）

**比值 0.33 且八句都在 0.27~0.47 → 穩定的系統性現象，不是隨機。**
時間覆蓋率換算：375 位置涵蓋 30 秒 → 每位置 80ms；第一句長 13.375s → 167 個位置含語音，
只保留 27 個 → **僅用了 16% 的語音訊息，84% 被丟棄**。

### 生成品質（`--mode nobias`，prompt 完全不給 entity）

```
REF: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
HYP: mister quill   is the apostle of the churches      and we are glad to hear him say so

REF: nor is mister quilter's manner less interesting than his matter
HYP: nor is it the case that mister cartier's manner is so interesting as to be inter...

REF: he tells us that at this festive season of the year with christmas and roast beef...
HYP: he tells us that at this festive season of the year there is christmas in the mo...
```

> **結論 1：模型確實在使用音訊。** 第三句**前 11 個字一字不差**，
> 第一句連 `apostle` 這個 rare word 都正確 —— **而且 prompt 裡沒有任何 entity 提示**。
> 不聽音訊的模型不可能逐字產出 "at this festive season of the year"。
>
> 這回答了過擬合測試（ce → 0.0015）**證明不了**的問題：
> 過擬合只證明「架構可微分、能學」，不證明「音訊路徑被使用」。

> **結論 2：句子越往後越偏。** 開頭準確、中段開始飄、結尾變成「聽起來合理但錯誤」的文字。
> 與上述 ctc_compress 比值 0.33（84% 語音資訊被丟棄）直接對應。

**生成速度**：2.62 s/句 → dev-clean 2694 句約 2 小時（batch_size=1）

### WER（待填）

```
待跑：python src/evaluate.py --ckpt exp/stage1_tc100/best.pt --stack 4 \
          --manifest data/manifest/test-clean.tsv \
          --out exp/hyp/test-clean.stack4.nobias.tsv --mode nobias
     python fbai-speech/is21_deep_bias/score.py --refs ... --hyps ...
```

| 指標 | 我的（stack=4, 100h） | 論文（960h） |
|---|---|---|
| test-clean WER | 待填 | 4.54 |
| test-clean U-WER | 待填 | 2.34 |
| test-clean B-WER | 待填 | 23.39 |

---

## 實驗 2：Stage 1 ASR，stack=2（修正後）✅ 完成

**與實驗 1 的唯一差異**：`AudioAdapter(stack=2)`，可訓練參數 3,888,029 → **2,413,469**

輸出：`exp/stage1_tc100_s2/`　耗時 8.0 小時（479.2 分）

### 設定差異表

| 項目 | 實驗 1 | 實驗 2 |
|---|---|---|
| **AudioAdapter `stack`** | **4** | **2** |
| adapter 位置數 | 375 | 750 |
| 每秒位置數 | 12.5 | 25.0 |
| **CTC 無解句子比例** | **83.2%** | **0.00%** |
| adapter 參數量 | 2,949,696 | 1,475,136 |
| **可訓練參數總計** | 3,888,029 | **2,413,469** |
| 其餘設定 | 完全相同 | 完全相同 |

其餘設定：train-clean-100（100.59 h）、10 epochs、batch 4、CTC/CE 權重 0.3/1.0、
AdamW lr=1e-4 wd=0.01、linear warmup 3% + decay、grad clip 1.0、seed 42、
LoRA r=16 alpha=32 dropout=0.05 q_proj+v_proj、RTX 3090、encoder bf16

### ⭐ 結果對照

| 指標 | 實驗 1 (stack=4) | 實驗 2 (stack=2) | 改善 |
|---|---|---|---|
| **訓練 CTC loss 收斂值** | 2.70 | **0.775** | **−71%** |
| **最佳 val_total** | 2.0163 | **0.4246** | **−79%** |
| 最佳 epoch | 6 | 5 | |
| **之後為何停滯** | **欠擬合**（83% 資料對 CTC 無效） | **過擬合**（資料量不足） | |
| `ctc_compress` 比值 | 0.33 | 待測 | |
| 耗時 | 8.5 h | 8.0 h | |

**只改一個參數，validation loss 降了近八成。診斷完全成立。**

### 完整學習曲線（`exp/stage1_tc100_s2/epoch_summary.tsv`）

| epoch | tr_total | tr_ctc | tr_ce | val_total | val_ctc | val_ce | lr | 累計分 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.5032 | 1.7011 | 1.9929 | 0.5892 | 0.9821 | 0.2946 | 9.28e-05 | 49.0 |
| 1 | 0.4535 | 0.8440 | 0.2003 | 0.4882 | 0.9195 | 0.2123 | 8.25e-05 | 98.5 |
| 2 | 0.3837 | 0.8125 | 0.1400 | 0.4510 | 0.9032 | 0.1801 | 7.22e-05 | 147.5 |
| 3 | 0.3510 | 0.8003 | 0.1109 | 0.4385 | 0.8980 | 0.1691 | 6.19e-05 | 194.9 |
| 4 | 0.3295 | 0.7931 | 0.0915 | 0.4284 | 0.8939 | 0.1602 | 5.15e-05 | 242.8 |
| **5** | 0.3133 | 0.7882 | 0.0769 | **0.4246** | 0.8891 | 0.1579 | 4.12e-05 | 290.1 |
| 6 | 0.2998 | 0.7841 | 0.0646 | 0.4249 ↑ | 0.8911 | 0.1576 | 3.09e-05 | 337.8 |
| 7 | 0.2896 | 0.7807 | 0.0553 | 0.4316 ↑ | 0.8888 | 0.1650 | 2.06e-05 | 384.9 |
| 8 | 0.2810 | 0.7776 | 0.0477 | 0.4375 ↑ | 0.8849 | 0.1720 | 1.03e-05 | 432.3 |
| 9 | 0.2738 | 0.7748 | 0.0413 | 0.4426 ↑ | 0.8834 | 0.1775 | 0.00e+00 | 479.2 |

### 三個觀察

**1. epoch 0 → 1 的大跳躍**（tr_total 2.50 → 0.45）
epoch 0 的平均包含最初那些高 loss 的步數，所以偏高。不是異常。

**2. epoch 5 是轉折點 — 教科書式的過擬合**
```
訓練 loss 持續下降：0.3133 → 0.2738
驗證 loss 開始回升：0.4246 → 0.4426
```

**3. 過擬合發生在 CE 不在 CTC**
```
val_ce :  0.1579 → 0.1775   明顯回升，過擬合
val_ctc:  0.8891 → 0.8834   持續微降，沒有過擬合
```
合理 —— SmolLM2 + LoRA 有更多容量去死記訓練資料，而 CTC 那側只有 adapter + 一層 Linear。

### ⭐ 兩次「都在第 6 個 epoch 停滯」但原因完全相反

| | 實驗 1 | 實驗 2 |
|---|---|---|
| 最佳 epoch | 6 | 5 |
| 原因 | **欠擬合** — 83% 資料對 CTC 無效 | **過擬合** — 100h 資料量不足 |
| 症狀 | train 和 val 都降不下去 | train 持續降、val 回升 |

→ **這反過來支持「論文的超參數是自我一致的」**：論文的 10 epochs 是針對 **960 小時**校準的。資料量放大 9.6 倍後，過擬合會延後，10 epochs 才是合理設定。

### 待填

- [ ] `ctc_compress` 比值（改 `src/test.py` 的 `CKPT` 和 `stack=2`）
- [ ] 生成品質對照（看後段飄移是否減輕）
- [ ] test-clean WER

---

## ⚠️ evaluate.py 的兩個 bug（2026-10-12 發現）

跑 `score.py` 時失敗，三個數字對不上暴露了兩個獨立問題：

```
hyp 檔行數          2629
score.py 讀到        2624 筆
Dataset 載入         2611 句
```

### Bug A：生成的文字含換行符，破壞 TSV 格式

`2629 行 > 2611 句`，多出 18 行 —— 模型生成的文字裡有 `\n`，寫進 TSV 就把一句拆成多行。

**修法**：回傳前正規化所有空白字元。

```python
text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
return " ".join(text.split())      # \n \t 多重空白全部收成單一空格
```

`" ".join(text.split())` 比 `.replace("\n", " ")` 穩健 —— `split()` 不給參數會按任何空白切並忽略空字串。

### Bug B：評估時不該過濾超長句（觀念問題，非只是 code）

```
ValueError: 908-157963-0007 missing in hyps!
```

那 9 句是 `duration > 30` 被過濾掉的。

**訓練時過濾是對的**（不要教模型輸出它沒聽到的字），**但評估時不該過濾** —— 測試集是固定的，少測 9 句 WER 就跟論文不可比。而且論文的模型也會在 30 秒截斷，大家條件一樣。

**修法**：

```python
ds = LibriSpeechDataset(args.manifest, fe, max_duration=1e9)
```

### 暫時解法

`score.py` 有 `--lenient` 可以忽略缺少的 utterance，但**缺的句子會算全錯 → 數字偏高**，只能當粗估，不可寫進報告。

---

## 待辦

### 🔴 立即（上次中斷在這裡）

- [ ] `python src/dump_tb.py exp/stage1_tc100` 還原實驗 1 的學習曲線
- [ ] 修 `evaluate.py` 的兩個 bug（換行符、評估時誤用訓練用的超長句過濾）
- [ ] 測 stack=2 的 `ctc_compress` 比值 ← **核心發現缺的最後一塊證據**（stack=4 是 0.33，應往 1.0 靠近）
- [ ] 兩組完整 test-clean WER（修完 bug 後，各 2 小時，可雙卡並行）
- [ ] 新模型的生成品質對照（看後段飄移是否減輕）

### 之後

- [ ] fp32 vs bf16 對照（論文未指定精度，需驗證不影響收斂）
- [ ] `projector.py`（Eq. 1、2）
- [ ] `losses.py`（Eq. 3、4、5-7）← 核心貢獻
- [ ] `train_stage2.py`（M=120、batch 1、7 epochs、121 次 forward + KV cache）
- [ ] `bti.py`（top-K=10 + `<unbiased>` 門檻）/ `recall.py`（Recall@X、Recall#X）
- [ ] 四組 loss 對照實驗（教授交辦的學習曲線）
- [ ] 考慮上 960h（約 2.9 天）

---

## 向教授請示的事項

1. **能否向第一作者詢問未公開的設定**？目前已靠參數預算與資料分布反推出 CTC 詞彙（29）、
   LoRA rank（16）、adapter 降採樣率（stack ≤ 2），但 lr / optimizer / projector 的 D′
   仍是自選。
2. **能否取得作者的 N=5000 biasing list 或其 seed**？官方 ref 只有 N=100/500/1000/2000，
   論文 Table 1 整表用 N=5000 → 否則無法逐句對齊。
3. **訓練規模**：目前用 train-clean-100（100h / 8.5 小時）。要不要上 960h（約 2.9 天）？
4. **Stage 2 的 121 次 forward**：論文 instruction 為單數、M=120、batch=1，推斷每個候選
   各跑一次 forward。`seg1 + audio` 約 105 token 對 121 次完全相同，必須用 KV cache
   （`past_key_values`），否則訓練時間不可接受。**論文未提此處如何處理**，想確認作者做法。
