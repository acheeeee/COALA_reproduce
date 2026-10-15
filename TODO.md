# COALA Reproduce — 資源清單與進度追蹤

論文：arXiv:2607.08117（INTERSPEECH 2026, NTNU）
LaTeX 原始檔：`paper/src/main.tex`　圖片：`paper/src/images/*.png`

---

## 1. 模型資源（要拿現成的 vs 要自己訓練）

### ✅ 已實測確認的模型規格（`src/check_models.py`，2026-09-18）

| 項目 | 實測值 | 論文說 | 備註 |
|---|---|---|---|
| Whisper-large-v2 **encoder** 參數量 | **636.8M** | 640M | ✅ 吻合（論文四捨五入） |
| 整個 `WhisperModel`（含 decoder） | 1543.3M | — | ⚠️ decoder 白佔 906M，下載 6.17GB 大半用不到，上 GPU 前要處理 |
| Whisper encoder `d_model` | **1280** | 未寫 | → audio adapter 的**輸入**維度 |
| Whisper encoder 層數 | 32 | 未寫 | |
| SmolLM2-135M-Instruct 參數量 | **134.5M** | 135M | ✅ 吻合 |
| SmolLM2 `hidden_size` | **576** | 未寫 | ⭐ **這就是 Eq. 2 的 `D`**，projector 輸入維度 |
| SmolLM2 `vocab_size` | 49152 | 未寫 | |
| SmolLM2 層數 | 30 | 未寫 | → LoRA 掛在這 30 層上 |
| encoder + LM 合計 | 771.3M | 777M | **差額 5.7M = 我要自己訓練的全部元件** |

**5.7M 參數預算可以用來反推論文沒寫的設定**（疑點 #3~#6）：audio adapter + CTC module + discriminative projector + 兩組 LoRA 全部加起來只能約 5.7M。這個上限對 CTC 的輸出詞彙大小限制特別強。

| 元件 | 來源 | 參數量 | 拿現成 / 自己訓練 | 備註 |
|---|---|---|---|---|
| Audio Encoder | `openai/whisper-large-v2` 的 **encoder 部分** | 640M | **拿現成，全程 frozen** | 只用 encoder，decoder 不要 |
| Backbone LM | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135M | **拿現成，權重 frozen** | 只透過 LoRA 微調 |
| Audio Adapter | 無現成 | ? | **自己訓練（from scratch）** | 論文沒給架構細節 → 待決定 |
| CTC Module | 無現成 | ? | **自己訓練（from scratch）** | 接在 audio adapter 之後 |
| LoRA (ASR) | 無現成 | ? | **Stage 1 自己訓練** | rank / alpha / target modules 論文未給 |
| LoRA (Scoring) | 無現成 | ? | **Stage 2 自己訓練** | ❓待確認：與 Stage 1 的 LoRA 是不是同一組（看 `images/algorithm.png` 圖 (a) 的 Backbone LM 方塊內部） |
| Discriminative Projector | 無現成 | ~2 層 MLP | **Stage 2 自己訓練** | D×D′ → D′×1，論文未給 D′ |

總參數量論文寫 777M（= 640M + 135M + adapter/CTC/LoRA/projector）。

## 2. 資料資源

| 項目 | 來源 | 大小 | 狀態 |
|---|---|---|---|
| LibriSpeech train-clean-100 / train-clean-360 / train-other-500 | OpenSLR | ~60GB | ✅ 已下載 `bedo@79:/mnt/disk2/LIBRISPEECH`（合計 = 論文要的 960h）|
| LibriSpeech dev-clean / dev-other | 同上 | ~340MB each | ✅ 已下載 |
| LibriSpeech test-clean / test-other | 同上 | ~350MB each | ✅ 已下載 |
| ☐ 解壓縮確認 | — | — | `LibriSpeech/` 目錄已存在，但**尚未確認 7 個 tar 是否都解開** |
| Rare word list | [fbai-speech `is21_deep_bias/words/`](https://github.com/facebookresearch/fbai-speech/tree/main/is21_deep_bias) | ~4MB | ☐ 未取得 |
| 官方評估腳本 `score.py`（U-WER / B-WER） | 同上 repo 根目錄 | 小 | ☐ 未取得 — **一定要用官方的，否則數字不可比** |
| 官方 test biasing list `ref/*.tsv` | 同上 `is21_deep_bias/ref/` | 小 | ☐ 未取得。**只有 N=100/500/1000/2000** |
| Biasing list **N=5000** | ⚠️ 官方沒有，要自己產生 | — | ☐ 需自己寫產生器 |

### ⚠️ 重要落差：N=5000 的 biasing list 無法完全對齊

- 論文 Table 1（Recall 全表）用 N=5000，Table 2 也有 N=5000 欄位。
- 但 Le et al. 官方 ref 檔只提供 N=100/500/1000/2000 → 作者的 N=5000 是自行隨機採樣，**random seed 未公開**。
- 影響：N=500 / N=1000 可與論文逐句對齊；**N=5000 的數字必然有隨機差異**，不能當成 reproduce 失敗。
- 對策：自己產生時固定 seed 並記錄，同一份 list 重複使用。這條寫進 thesis limitation。

## 3. 環境 / 硬體

- 論文原始：**單張 24GB RTX 3090**（N=5000 不做 BTI 會 OOM，這個 OOM 本身是論文結果之一）
- 伺服器 `bedo@79`，我的工作目錄 `/mnt/disk2/M11515125/`

| GPU | 型號 | VRAM | 用途 |
|---|---|---|---|
| 0 | RTX A5000 | 24564 MiB | 開發 / 平行實驗 |
| **1** | **RTX 3090** | **24576 MiB** | ⭐ 與論文同款同容量 → 對齊論文的實驗都用這張 |
| 2, 3 | RTX 2080 Ti | 11264 MiB | ❌ 太小不用 |

- Driver 535.129.03 / CUDA 12.2 → PyTorch 用 cu121
- 系統 Python 3.8.10（Ubuntu 20.04，太舊且無 root 不能升 → 所以用 conda 自備 Python）
- Miniconda 安裝位置：`/mnt/disk2/M11515125/miniconda3`（conda 26.7.1），已 `conda init bash`（改過 `~/.bashrc`）
- 磁碟：`/mnt/disk2` 共 11T，可用 4.1T → 空間無虞
- 專案環境：`conda activate coala`（Python 3.10）
- ✅ **好消息：有跟論文完全相同的 24GB RTX 3090，所以 Table 2「N=5000 → OOM」這格可以忠實重現**（多數 reproduce 做不到這點，值得寫進 thesis）
- ⚠️ 共用機器，跑任何東西前先 `nvidia-smi` 看別人有沒有在用，並且一律 `CUDA_VISIBLE_DEVICES=1` 指定

## 4. 官方參考實作

- https://github.com/Guo0911/COALA （論文 footnote 1）
- 定位：**對照驗證**用，不是照抄。先自己實作 → 卡住或數字對不上再開來比對
- 第一作者 Jhih-Rong Guo 是同實驗室 → 可以直接問

---

## 5. 論文沒寫清楚的地方（疑點清單，之後寫進 thesis）

### ✅ B 層第 4 項完成 + GPU 煙霧測試 — 2026-10-07

**已實作模組**（`src/modules.py`、`src/dataset.py`）：

| 模組 | 規格 | 參數量 |
|---|---|---|
| `AudioAdapter` | (B,1500,1280) → (B,375,576)，frame stacking k=4 + Linear | 2,949,696 |
| `CTCHead` | (B,375,576) → (B,375,29) | 16,733 |
| `ctc_compress` | (375,576) → (L',576)，去 blank + 去連續重複 | 0 |
| **合計** | | **2,966,429 / 5.7M 預算** |

**剩餘預算推論**：5.7M − 2.97M = 2.73M；再扣 projector（Linear(576,576)+Linear(576,1) ≈ 333K）→ 約 **2.4M 給兩組 LoRA**，每組 1.2M。SmolLM2 有 30 層、hidden 576，LoRA 掛 q_proj/v_proj 每層約 `r×1920`，30 層 `r×57,600` → **推得 LoRA rank r ≈ 16~20**（論文未寫，見疑點 #3）。

**CTC 詞彙已用實際資料確認**：`awk ... | fold -w1 | sort -u` 掃過 train-clean-100 的所有 text，只出現 `空格 + 撇號 + a-z` = 28 個字元，無數字無其他標點 → **加 blank 剛好 29** ✅

**Dataset 驗證**：dev-clean 載入 2694 句、跳過 9 句（>30s）。第一句 `ctc_ids=(89,)` 與逐字稿「73 字母+16 空格」手算吻合。`collate_fn` padding 正確（`ctc_lens=[89,63,172,133]` → pad 到 172）。

**GPU 實測（RTX 3090, batch=4, fp32）**：

```
encoder 上 GPU          2.43 GB
峰值（前向跑完）         2.80 GB   → 24GB 還剩 21GB
壓縮：375 → 158/172/219/196 tokens（隨機初始化）
```

→ **VRAM 充裕，不需 fp16。**
→ **檢查指標**：訓練後壓縮結果應接近實際字元數（第一句應為 ~89）。目前 158，可用來判斷 CTC 是否真的學到對齊。

### ✅ B 層第 5 項子任務 a~d 完成：Stage 1 模型組裝 — 2026-10-10

**檔案**：`src/prompt.py`（prompt 組裝 + pad_batch）、`src/lora_setup.py`、`src/model.py`（COALAModel）

**完整資料流已驗證可跑**：
```
input_features (B,80,3000)
  → Whisper encoder [凍結, no_grad] → (B,1500,1280)
  → AudioAdapter → (B,375,576) ──┬→ CTCHead → (B,375,29) → CTC loss (w=0.3)
                                 ↓ ctc_compress 逐句（長度不同）
                            build_stage1_sample 逐句
                                 ↓ pad_batch
                            (B,max_tot,576) → SmolLM2+LoRA("asr") → CE loss (w=1.0)
total = 0.3·ctc + 1.0·ce
```

**驗證結果（dev-clean, batch=2, RTX 3090）**：

| 項目 | 實測 | 預期 | 判定 |
|---|---|---|---|
| 可訓練參數 | **3,888,029** | 2,949,696+16,733+921,600 = 3,888,029 | ✅ **精準命中** |
| 有梯度的張量 | **124/124** | adapter 2 + ctc 2 + LoRA 30層×2模組×2矩陣=120 | ✅ 整條鏈可微分 |
| ctc_loss | 13.39 | (T·lnC − ln#paths)/S，S=89→11.9、S=63→17.3，均 14.6 | ✅ |
| ce_loss | **5.60** | ≪ ln(49152)=10.80 | ✅ **證明預訓練權重確實載入** |

**三個重要觀念記錄：**

1. **梯度可以穿過 argmax 的「選擇」動作**。`ctc_compress` 用 argmax 決定保留哪些 frame，argmax 本身不可微分，但梯度沿著被選中的 `hidden` 流回去——選擇不需要梯度，被選中的內容需要。
2. **預訓練模型的初始 CE loss 應明顯低於 ln(vocab)**。若 `ce_loss ≈ 10.80` 代表 LM 權重沒載入（變成隨機初始化）。這是權重載入正確性的檢查手法。
3. **凍結與包 peft 的順序不能反**。`get_peft_model` 本身會自動凍結非 LoRA 參數；若先包再把全部參數設 `requires_grad=False`，會把剛建好的 LoRA 一起凍結 → 可訓練參數變 0、loss 完全不降、且不報錯。

**CTC loss 的三個介面陷阱（`compute_ctc_loss`）**：
- 要 `(T, B, C)` 維度順序（與 PyTorch 慣例相反）→ 需 `transpose(0,1)`
- 吃 **log 機率**不是 logits → 要先 `log_softmax(dim=-1)`。忘記不會報錯，只會算出錯的數字
- `target_lengths` 傳錯不會報錯，只會把 padding 的 0（= blank id）當成真實目標去學 → 靜默失敗
- `zero_infinity=True`：防止單筆 inf 毀掉整個 batch 的梯度

### ✅ 子任務 5e：過擬合測試通過 — 2026-10-10

`src/overfit_test.py`：固定 2 句（dev-clean 前兩筆）重複訓練 200 步，AdamW lr=1e-4。

```
step    1  total  9.5699  ctc 12.7989  ce  5.7302
step   20  total  2.1226  ctc  3.3141  ce  1.1283
step   60  total  0.8974  ctc  2.7017  ce  0.0869
step  100  total  0.6533  ctc  2.1443  ce  0.0100
step  180  total  0.3886  ctc  1.2898  ce  0.0017
step  200  total  0.3519  ctc  1.1343  ce  0.0116
```

**判定 ✅**：`ce 5.73 → 0.0017`（完全死記 2 句）證明梯度路徑、labels 對位、loss 權重全部正確。`ctc 12.80 → 1.13` 且仍在下降 — CTC 降得慢屬正常（adapter + CTC head 從零訓練，CE 那側有預訓練 LM 幫忙）。

**⚠️ 這個測試證明不了的事（待後續確認）**：

`ce_loss → 0` **不保證模型在用音訊**。prompt 裡的 entity 清單本身含部分 rare word，且同一 batch 重複 200 次，LM 可能學成「看到這組 entity 清單就背出這句話」而完全忽略 audio token。

→ **真正的驗證是 dev-clean 上的 WER**（未見過的音訊）。若 WER 合理則音訊路徑確實被使用。
→ 另一個可選的直接檢查：過擬合後把 audio token 歸零，看 `ce_loss` 是否跳高。若不變則音訊沒被用到。

**訓練三步式**（之後所有訓練迴圈的骨架）：
```python
opt.zero_grad()    # PyTorch 梯度是累加的，不清掉會越來越大然後爆掉
loss.backward()
opt.step()
```

**⚠️ 凍結模組的 train/eval 模式**：`model.train()` 會**遞迴**套用到所有子模組，包含凍結的 Whisper encoder → 會啟用 dropout → 同樣音訊每步輸出不同 → adapter 在學一個抖動的目標，過擬合測試降不到 0。

**通則：每次 `.train()` 之後要把凍結的子模組設回 `.eval()`。**
```python
model.train()
model.encoder.eval()
```

### ⚡ 效能瓶頸分析與 bf16 優化 — 2026-10-10

**問題**：首次煙霧測試實測 **1.04 s/it**（batch=4），推算 train-960 × 10 epochs = **8.5 天**，共用伺服器上不可行。

**profiling 結果**（`src/profile_step.py`，batch=4, RTX 3090）：

```
完整訓練步              924.4 ms
  encoder (fp32)        759.4 ms   ← 佔 82.1%
  encoder (bf16)        193.7 ms   ← 加速 3.92x
  其他（adapter/CTC/prompt/LM fwd+bwd）  165.0 ms

不同 batch size 的每句成本（encoder fp32）:
  bs= 4:  193.9 ms/句, 峰值 3.37 GB
  bs= 8:  183.4 ms/句, 峰值 3.72 GB
  bs=16:  192.5 ms/句, 峰值 4.41 GB
```

**三個結論：**

1. **Whisper encoder 佔 82%** — 它凍結卻每 epoch 重算，是唯一值得優化的地方
2. **bf16 加速 3.92x**（遠超預期的 2x）。原因：**RTX 3090 的 tensor core 支援 bf16 但不支援 fp32**，用 fp32 等於放著專用硬體不用。這不是精度換速度，是本來就該走的路徑
3. **加大 batch 完全無效**（每句成本幾乎不變）→ GPU 在 bs=4 已算到飽。**好消息：論文指定的 batch=4 不造成速度劣勢，可忠實照論文設定跑**

**bf16 數值驗證**（凍結 encoder 的輸出）：

```
fp32 範圍 [-11.961, 28.055]  std 0.8751
bf16 範圍 [-11.958, 28.023]  std 0.8746
MAE 0.005779  →  MAE/std = 0.66%（符合 bf16 理論精度 2⁻⁸≈0.39% + 32 層累積）
餘弦相似度 0.999476
過擬合測試（改 bf16 後）ce 5.73 → 0.0015、ctc 12.80 → 1.075  ← 與 fp32 幾乎相同
```

⚠️ **指標選擇的教訓**：逐元素相對誤差 `|a−b|/(|a|+eps)` 算出 10.27%，看起來很糟，但那是**被接近零的小值主導**的壞指標（std 0.875，大量元素接近 0）。驗證數值差異應該用「MAE / std」或餘弦相似度。

**採用的寫法**：

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = self.encoder(input_features)
x = out.last_hidden_state.float()     # 必須轉回 fp32 給可訓練的 adapter

h = self.adapter(x)                   # ⚠️ 必須在 with 外面
ctc_logits = self.ctc_head(h)         # ⚠️ 必須在 with 外面
```

- 在 `no_grad` 裡 → adapter / CTC head **拿不到梯度**，永遠學不到（但 loss 仍會降因為 LoRA 在學，極難察覺）
- 在 `autocast` 裡 → 可訓練 Linear 用 bf16 算，梯度精度受損
- `.float()` 放 with 內或外無功能差異（autocast 不覆寫明確型別轉換），放外面只是更清楚

**修正後的時間預算**（每步 358.7 ms）：

| 資料集 | 原本 | bf16 後 |
|---|---|---|
| train-clean-100 × 10 ep | 21 小時 | **7.1 小時** |
| train-960 × 10 ep | 8.5 天 | **2.9 天** |

**未採用的備案（若 2.9 天仍太久）**：預先算好 encoder 輸出存檔。每句 (1500,1280) fp16 = 3.84 MB → tc100 需 110 GB、960h 需 1.08 TB（`/mnt/disk2` 剩 4.1 TB，可行）。預估可再加速至 1.3 天。代價是一次性生成時間 + 無法做音訊資料增強（論文未用）。

### ⚠️ checkpoint 只存可訓練參數的注意事項

`save_ckpt` 只存 3.9M 可訓練參數（約 15MB），不存凍結的 encoder/LM（省 2.5GB）。載入必須用 `strict=False`，否則會因缺少幾千個凍結參數的鍵而報錯。

**但 `strict=False` 會靜默忽略拼錯的鍵** → 若改過模組名稱，載入會「成功」但權重沒進去、模型變回隨機初始化。防護：

```python
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
assert len(unexpected) == 0, f"checkpoint 有不認識的鍵: {unexpected}"
```

`unexpected`（checkpoint 有但模型沒有的鍵）**必須為空**。




```
RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16
```

**v4**：`from_pretrained` 一律轉成 fp32。
**v5**：沿用 checkpoint 自己的 dtype → SmolLM2 載成 **bf16**，Whisper 載成 **fp32**。

這會在完整 pipeline 再撞一次：Whisper(fp32) → AudioAdapter(fp32, nn.Linear 預設) → SmolLM2(**bf16**) ✗

**決定：全專案統一 fp32**（VRAM 還剩 21GB，不缺；fp32 無數值風險。之後要加速再考慮 AMP）

```python
AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32)
```

**通則：所有 `from_pretrained` 都明確指定 dtype，不依賴預設。**
（transformers 5.x 參數名是 `dtype`，v4 是 `torch_dtype`）

### ⚠️ 踩過的坑：函式內新建 tensor 要指定 device

`torch.tensor([True])` 預設建在 CPU → 上 GPU 後 `torch.cat([cpu, cuda])` 直接報錯。CPU 測試時完全看不出來。

**通則：函式內部新建 tensor 一律寫 `device=某個輸入.device`。**

```python
first = torch.ones(1, dtype=torch.bool, device=pred.device)
```

**未處理的邊界情況**：若某句 CTC 預測全是 blank，`keep` 全為 False → 回傳 `(0, 576)` 空 tensor。訓練初期可能發生，之後要處理（至少保留 1 個 token）。

### ⚠️ DataLoader 效能

`__getitem__` 每次都要讀 flac + 算 log-Mel（3000 frame 的 FFT，純 CPU 重活）。28 萬句 × 10 epoch = 280 萬次。

→ 必須設 `num_workers=8`（背景預取）+ `pin_memory=True`（加速 CPU→GPU 傳輸）。

### ✅ 已實測的 shape 鏈條（`src/check_shape.py`，2026-10-01）

```
波形 (16000 Hz)  →  log-Mel (1, 80, 3000)  →  encoder (1, 1500, 1280)
```

| 測試音檔 | 時長 | log-Mel shape | encoder 輸出 shape |
|---|---|---|---|
| 3081-166546-0073 | 1.445 s | (1, 80, 3000) | (1, 1500, 1280) |
| 422-122949-0013 | 32.645 s | (1, 80, 3000) | (1, 1500, 1280) |

→ **Whisper 一律 pad/截斷到固定 30 秒**（3000 frames = 30s × 100 fps），encoder 的 conv stride=2 把 3000 對折成 **T = 1500**。

**Audio adapter 的規格因此確定：輸入 `(B, 1500, 1280)` → 輸出 `(B, ?, 576)`。**

### ⚠️ 關鍵推論：adapter 必須降採樣，不只是換維度

SmolLM2-135M 的 context window 只有 **2048 token**：

```
1500 audio token + ~30 instruction + ~5 entity + ~40 transcription ≈ 1575
```

單次勉強可行，但 **Stage 2 一個 utterance 要跑 121 次 forward**，每次都帶 1500 個 audio token → 計算上不可行。**這正是論文 Table 2「N=5000 → OOM」的機制。**

→ 論文 3.4「CTC module generates frame-level alignments that are projected into the backbone LM embedding space as audio tokens」的真正意義：**用 CTC 預測結果濾掉 blank 與重複 frame（CTC compression）**，把 1500 壓到與實際字數同量級（約數十個）。

1500 frames 對應約 40 個字 → 平均一字佔 37 frames，其餘全是 blank/重複，壓縮空間極大。

**這是 adapter + CTC 設計的核心，論文完全沒寫（見疑點 #5、#6），需自行決定並記錄。**

### ⚠️ 超過 30 秒的音檔會被靜默截斷

實測最長的 dev-clean 句子為 32.645 秒 → Whisper 只處理前 30 秒，**後 2.6 秒的聲音消失但逐字稿完整**，模型被要求輸出沒聽到的字。這種資料會污染訓練。

對策：manifest 已有 `duration` 欄位，訓練時過濾 `duration > 30`。需統計受影響句數並記為 limitation。

### ✅ 已確定的資料格式決策（2026-09-21）

**文字正規化方向：全部轉小寫。**

| 來源 | 大小寫 |
|---|---|
| LibriSpeech `*.trans.txt` | **全大寫**（唯一的異類） |
| 官方 `ref/*.tsv` 參考文字 | 全小寫 |
| 官方 `ref/*.tsv` biasing list | 全小寫 |
| `all_rare_words.txt` / `common_words_5k.txt` | 全小寫 |

→ `score.py` 比對的是小寫參考文字，所以 **逐字稿要 `.lower()`**，ASR 輸出也必須是小寫無標點。不要反過來把 word list 轉大寫。

**LibriSpeech 目錄與檔名格式：**
```
LibriSpeech/<split>/<speaker_id>/<chapter_id>/<speaker>-<chapter>-<utt>.flac
                                             /<speaker>-<chapter>.trans.txt
```
- `trans.txt` 每行 = `utt_id` + 一個空格 + 整句逐字稿（用 `split(" ", 1)` 解析）
- 逐字稿全大寫、無標點，唯一例外是所有格撇號（`QUILTER'S`）

**官方 `ref/*.tsv` 四欄格式（tab 分隔）：**
1. utterance ID
2. 參考文字（小寫）
3. **句中的 rare words = E⁺**（可能是 `[]`，代表這句沒有 rare word → 此時 `<unbiased>` 是唯一正樣本）
4. **完整 biasing list = E**（含 distractors，是第 3 欄的超集）

第 3、4 欄都是 **JSON 陣列**（用 `json.loads()` 解析），且**照字母排序**（不是出現順序 → 正樣本位置不會洩漏資訊）。

### ⚠️ 踩過的坑：rare word 的判定方式（2026-09-30 修正）

**錯誤做法**：用 `word in all_rare_words.txt` 判定是否為 rare word。
→ 結果只有 **2512/2620** 列與官方一致，漏掉 `whittaws`、`craswellers`、`andella`、`thel`、`beeder` 這類詞。

**原因**：`all_rare_words.txt` 是從 **LibriSpeech 訓練語料**的詞彙推導出來的。只出現在 test set、訓練時從未見過的 OOV 詞根本不在裡面。

**正確做法**（官方 README 明載）：

| 檔案 | 正確用途 |
|---|---|
| `common_words_5k.txt` | **判定 rare word** → 用補集：「不在這 5000 常見詞裡的字」都算 rare（OOV 自然包含） |
| `all_rare_words.txt` | **只用來採樣干擾詞 E⁻** 的詞池 |

修正後 → **2620/2620 完全一致** ✅

### 官方 biasing list 長度不是固定值（已釐清，可忽略）

`len(list) - len(positives)` 的分布：500→1713、499→672、498→195、497→34、496→5、495→1（平均缺 0.45）。

這是**有放回抽樣後去重**的特徵（期望重複數 ≈ 500²/(2×209291) ≈ 0.6，與實測同量級）。

**對策：test set 評估一律直接用官方 `ref/*.tsv`，不自己產生**，這樣 N=500/1000 逐句對齊官方、零隨機性差異。自產的產生器只用於：

| 用途 | 大小 | 來源 |
|---|---|---|
| Stage 1 訓練 prompt | 10 | 自產（每 epoch 重抽） |
| Stage 2 訓練候選清單 | M=120 | 自產 |
| N=5000 評估 | 5000 | 自產（官方無，已知 limitation） |
| N=500 / 1000 評估 | — | **用官方 ref 檔** |

### 我的實際環境版本（2026-09-18 鎖定，見 `requirements.txt`，共 63 個套件）

| 套件 | 我的版本 | 備註 |
|---|---|---|
| torch | 2.5.1+cu121 | |
| torchaudio | 2.5.1+cu121 | |
| **transformers** | **5.17.0** | ⚠️ **風險：論文是 2026-07 發表，當時主流是 4.x。v4→v5 有破壞性 API 變更。先用 v5 試，撞牆再降版** |
| peft | 0.21.0 | LoRA（含多 adapter 切換） |
| accelerate | 1.15.0 | |
| numpy | 2.2.6 | |
| Python | 3.10.21 | 論文未指定 |

已確認整份 `main.tex` 內都沒有出現的資訊：

1. **learning rate / optimizer / scheduler / warmup** — 完全沒寫
2. **random seed** — 沒寫
3. **LoRA 設定** — rank、alpha、dropout、target modules 都沒寫
   → **✅ 已由參數預算反推並實測確認（2026-10-08）**
   ```
   推算：5.7M − adapter/CTC 2.97M − projector 0.33M ≈ 2.4M → 兩組各 1.2M
        SmolLM2: 30 層, hidden 576, 9 heads(head_dim 64), 3 KV heads(GQA)
        q_proj 576→576: 16×576 + 576×16 = 18,432
        v_proj 576→192: 16×576 + 192×16 = 12,288
        每層 30,720 × 30 層 = 921,600
   實測：trainable params: 921,600  ← 與推算完全一致 ✅
   ```
   **採用設定**：`r=16, alpha=32, dropout=0.05, target_modules=["q_proj","v_proj"], bias="none"`
   - `q_proj`+`v_proj` 的依據：LoRA 原論文 Table 5，固定預算下 Wq+Wv 優於只調 Wq 或調全部四個
   - `alpha/r = 2`：LoRA 更新量為 `(alpha/r)·BA`，維持此比值則改 r 時不需重調 lr
   - 兩組具名 adapter（`asr` / `scoring`）共存驗證通過：`set_adapter()` 只啟用一組、另一組自動凍結，可訓練參數始終 921,600 → 符合論文 Stage 2「凍結舊參數、只訓練 scoring」的需求
4. **Discriminative projector 的 D′**（hidden dim）— 沒寫
5. **Audio adapter 架構** — 只說「audio adapter」，沒說是幾層 / conv 還是 linear / 降採樣率多少
6. **CTC module 細節** — output vocab 是 char 還是 BPE？怎麼把 frame-level alignment 投影成 audio token？
7. **Instruction prompt 完整文字** — 圖 (a) 只露出 "You are a ... Please determine if the text appears in the input speech"
   → **我的設計（`src/prompt.py`，2026-10-08）**：沿用 SmolLM2 的 chat template，四段串接
   ```
   seg1   = <|im_start|>system\nYou are a speech recognition assistant.<|im_end|>\n<|im_start|>user\n
   audio  = ctc_compress 輸出的 (L, 576) 向量，直接插在 user turn 開頭（對應論文圖 2a 的順序）
   seg2   = Transcribe the speech. The following entities may appear: {10 個 entity}.<|im_end|>\n<|im_start|>assistant\n
   target = {逐字稿}<|im_end|>
   ```
   - 把預設系統提示（"You are a helpful AI assistant named SmolLM..."，17 token）換成任務相關的短句
   - `labels`：seg1 / audio / seg2 全填 `-100`，只有 target 放真實 token id
   - 實測 token 數：文字 59 + audio 89 + target 12 = **160 / 2048**（用掉 8%）
   - **Stage 2 的共用前綴 = seg1 + audio ≈ 105 token**，對 121 次 forward 完全相同 → KV cache 可省 82% 重複計算
   - ⚠️ padding：訓練補右邊（labels 用 -100、mask 用 0 即可）；**生成時必須補左邊**，否則 causal LM 會從 padding 位置接續
8. ~~**Stage 2 每個 entity 是分開 forward 還是全部串成一個 sequence**~~
   → **已確認：分開 forward，一個 utterance = 121 次**（M+1）。證據：
   (a) 圖 (a) 的 instruction 寫 "determine if **the text**（單數）appears in the input speech"；
   (b) 圖 (a) 只有單一個 `E_1` 被餵進 tokenizer，不是 E_0..E_120 全部；
   (c) Stage 2 batch size 從 Stage 1 的 4 掉到 1——真正的負載是每個 utterance 內部展開的 121 條 sequence。
   → ⚠️ **衍生實作問題**：`[audio tokens][instruction]` 這段前綴對 121 次詢問完全相同，
     必須用 KV cache（HF 的 `past_key_values`）只算一次並重複使用，否則 Stage 2 會慢到跑不完。
     論文完全沒提這件事怎麼處理。
9. **Recall@X 的精確定義** — 「average number of highest-scoring entities required to achieve X% recall per utterance」，沒有 target entity 的句子怎麼算？

---

## 6. 進度日誌（每步都要記：改了什麼 → 得到什麼結果）

### 📌 教授交辦：要看學習曲線（2026-10-07）

**這不是附帶工作，而是驗證論文核心 claim 的唯一方法。**

論文 4.1 節的 claim 是過程性的，不是結果性的：
> baseline Bias-Loss 必須用 joint objective（disc + log loss）才能 **prevent training collapse**；MPD-Loss 與 DPD-Loss 可 **converge as standalone objectives**。

「collapse」和「converge」只能從 loss 曲線看出來，Table 1 的 Recall 數字看不出來。**而論文沒有提供這張圖**（只有 Figure 4 的靜態分數分布）→ 這會是我自己產出、論文沒有的證據。

**要跑的對照實驗**（Stage 2，同資料同超參數，只換 loss）：

| 實驗 | Loss | 預期 |
|---|---|---|
| A | discriminative only（**不加** aux log loss） | **崩潰** ← 論文沒跑這組，是關鍵對照 |
| B | discriminative + log loss (w=0.1) | 收斂（論文 baseline） |
| C | MPD-Loss 單獨 | 收斂 |
| D | DPD-Loss 單獨 | 收斂 |

四條曲線畫在同一張圖。A 崩潰而 C、D 平順 → claim 重現。

**要記錄的量**（tensorboard 已裝）：

Stage 1：`train/ctc_loss`、`train/ce_loss`、`train/total_loss`、`val/wer`、`train/lr`

Stage 2：`train/loss`、`val/recall_at_20`、`val/recall_at_50`、
**`train/pos_score_mean`、`train/neg_score_mean`** ← 這兩個直接對應論文 Figure 4 的三種行為（Bias-Loss 無法拉高正樣本分數／MPD 無法壓低負樣本／DPD 兩者皆可），逐 step 記錄可畫出**動態演化過程**，資訊量多於論文的靜態圖。

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("exp/stage2_dpd")
writer.add_scalar("train/loss", loss.item(), global_step)
# 看：tensorboard --logdir exp/
```


### ✅ A 層（資料層）完成 — 2026-09-30

**7 個 manifest 全部建好，句數與時長精準吻合官方統計：**

| split | 句數 | 時長 | 平均句長 | 無 rare word | 多目標(2+) |
|---|---|---|---|---|---|
| dev-clean | 2703 | 5.39 h | 7.2 s | 23.68% | 48.91% |
| dev-other | 2864 | 5.12 h | 6.4 s | 29.71% | 41.93% |
| test-clean | 2620 | 5.40 h | 7.4 s | 24.43% | 49.20% |
| test-other | 2939 | 5.34 h | 6.5 s | 27.15% | 44.37% |
| train-clean-100 | 28539 | 100.59 h | 12.7 s | 7.42% | **78.43%** |
| train-clean-360 | 104014 | 363.61 h | 12.6 s | 7.35% | **78.77%** |
| train-other-500 | 148688 | 496.86 h | 12.0 s | 10.01% | **73.29%** |
| **訓練集合計** | **281241** | **961.06 h** | | | |

### ⭐ 重要發現：訓練集的多目標比例遠高於評估集

訓練集 **73~79%** 的句子含 2 個以上 rare word，dev/test 只有 42~49%。原因是訓練句平均長 1.75 倍（12.6s vs 7.2s），字數多自然含更多 rare word。

→ **這是比論文 Figure 1 更強的動機支持**：訓練時有將近八成資料是多目標，全部受 discriminative loss 的 inter-positive competition 影響。這解釋了為何 baseline Bias-Loss 需要 auxiliary log loss（weight 0.1）才能收斂，而論文的 MPD/DPD-Loss 能獨立收斂。

### ✅ 已驗證：dev-clean 的 rare word 分布（重現論文 Figure 1，`src/figure1.py`）

```
rare words    utterances     ratio
----------------------------------
0                    640    23.68%
1                    741    27.41%
2-5                 1116    41.29%
6+                   206     7.62%
----------------------------------
total               2703   100.00%

最多的句子: 5338-24615-0002，含 24 個 rare word
少於 10 個的佔比: 98.30%   (論文 3.4 節說 99%)
```

**三個驗證結論：**

1. `total = 2703` 正好是官方句數，且與 `build_manifest.py` 的統計交叉一致 → 檔案遍歷與解析正確
2. **最大值 24 精準命中論文 Figure 4 的 `many (6-24)` 上界**。此數字是用**修正後的正確判定方式**算出（見上面「踩過的坑」），所以是真正的一致，非巧合
3. 多目標句佔 48.91%（dev-clean）／78% 以上（訓練集）→ 強力支持論文動機

**待追蹤：** 98.30% vs 論文 99%，差 0.7pp。論文那句是針對其資料而言，本次算 dev-clean，差異合理。

**訓練設計注意：** dev-clean 有 23.68% 的句子完全沒有 rare word（E⁺ 只有 `<unbiased>`），訓練集只有 7.4%。若 Stage 2 出現「模型把所有候選都判成負樣本」的傾向，這個比例是第一嫌疑犯。


| 日期 | Step | 做了什麼 | 結果 / 數字 | 備註 |
|---|---|---|---|---|
| 2026-09-16 | Step 0 | 確認論文、抓 LaTeX 原始檔到 `paper/` | — | 發現官方 code 存在 |
| 2026-09-16 | Step 1 | 第一次嘗試描述 biasing scoring data flow | ⚠️ 需要修正 | forward 次數的判斷理由不對 |
| 2026-09-16 | Step 1 | 從圖 (a) 推出「一個 entity 一次 forward，共 121 次」 | ✅ 過關 | 疑點 #8 結案；衍生出 KV cache 問題待研究 |
