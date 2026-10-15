---
tags: [COALA, 模型層, 檔案]
層: 模型層
狀態: 完成
---

# model.py

← [[模型層 總覽]] · [[COALA 索引]]

**組裝者。** 引用 [[modules.py]]、[[prompt.py]]、[[lora_setup.py]]。被 [[train_stage1.py]]、[[overfit_test.py]]、[[evaluate.py]] 使用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：組裝者 —— 把 [[encoder]]、[[AudioAdapter]]、[[CTCHead]]、[[lm]]（+ 兩組 LoRA）、[[tokenizer]] 組成 [[COALAModel]]，並實作 Stage 1 的 forward（CTC loss + CE loss）。

| 類型 | 內容 |
|---|---|
| import 外部 | `torch`、`torch.nn`、`peft`（`get_peft_model`）、`transformers`（`AutoModelForCausalLM`、`AutoTokenizer`、`WhisperModel`） |
| import 自己的 | [[lora_setup.py]]、[[modules.py]]、[[prompt.py]]；main 裡另外 import [[biasing.py]]、[[dataset.py]] |
| 被誰 import | [[train_stage1.py]]、[[evaluate.py]]、[[overfit_test.py]]（`COALAModel`、`WHISPER_ID`） |

**定義的函式 / 類別**

| 名稱 | 一句話 |
|---|---|
| [[COALAModel]] | 完整模型 |
| ↳ [[COALAModel.__init__()]] | 載入、組裝、凍結 |
| ↳ [[COALAModel.encode_audio()]] | 音訊 → [[h (adapter 輸出)]] + [[ctc_logits]] |
| ↳ [[COALAModel.forward_stage1()]] | batch → [[out (stage1 loss dict)]] |
| [[model.main()]] | 參數量 + 三個 loss + 124/124 梯度 |

**定義的常數**：[[WHISPER_ID]]、[[CTC_WEIGHT]]、[[CE_WEIGHT]]

**用到其他檔案的函式**：[[make_lora_config()]]、[[AudioAdapter]]、[[CTCHead]]、[[compute_ctc_loss()]]、[[ctc_compress()]]、[[build_stage1_sample()]]、[[pad_batch()]]

**資料流**：
```
batch → encode_audio → h (adapter 輸出), ctc_logits
  ├→ compute_ctc_loss → ctc_loss
  └→ 逐句 ctc_compress → build_stage1_sample → samples → pad_batch → padded → lm → ce_loss
loss (total) = 0.3·ctc + 1.0·ce
```
圖中物件：[[batch]]、[[h (adapter 輸出)]]、[[ctc_logits]]、[[ctc_loss]]、[[samples]]、[[padded]]、[[ce_loss]]、[[loss (total)]]
<!-- 結構:end -->

---

## 功能

把五個零件（Whisper encoder、AudioAdapter、CTCHead、SmolLM2、兩組 LoRA）組成完整模型。

常數：`WHISPER_ID`、`CTC_WEIGHT = 0.3`、`CE_WEIGHT = 1.0`（論文 3.4 Stage 1）

---

## `COALAModel(nn.Module)`

| 方法 | 輸入 | 輸出 |
|---|---|---|
| `__init__()` | 無 | — |
| `encode_audio(input_features)` | `(B, 80, 3000)` | `h (B, 375, 576)`, `ctc_logits (B, 375, 29)` |
| `forward_stage1(batch, entity_lists)` | `collate_fn` 的輸出、`list[list[str]]` | `dict(loss, ctc_loss, ce_loss)` |

### `__init__` 的四個部分

```python
# 1. Whisper encoder：凍結、eval、丟掉 decoder
whisper = WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32)
self.encoder = whisper.get_encoder().eval()
for p in self.encoder.parameters():
    p.requires_grad = False

# 2. 可訓練模組
self.adapter = AudioAdapter()
self.ctc_head = CTCHead()

# 3. LM：先凍結，再包 peft（順序不能反！）
self.lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
for p in self.lm.parameters():
    p.requires_grad = False
self.lm = get_peft_model(self.lm, make_lora_config(), adapter_name="asr")
self.lm.add_adapter("scoring", make_lora_config())

# 4. tokenizer
self.tokenizer = AutoTokenizer.from_pretrained(LM_ID)
```

---

## 完整資料流

```
input_features (B,80,3000)
  → Whisper encoder [凍結, no_grad, bf16] → (B,1500,1280)
  → AudioAdapter → (B,375,576) ──┬→ CTCHead → (B,375,29) → CTC loss (w=0.3)
                                 ↓ ctc_compress 逐句（長度不同）
                            build_stage1_sample 逐句
                                 ↓ pad_batch
                            (B,max_tot,576) → SmolLM2+LoRA("asr") → CE loss (w=1.0)

total = 0.3·ctc + 1.0·ce
```

**難點是中間那段「逐句處理後再 pad 回 batch」** —— 因為 `ctc_compress` 的輸出長度隨內容而異。

```python
samples = []
embed_layer = self.lm.get_input_embeddings()
for i in range(len(batch["text"])):
    audio_tok = ctc_compress(h[i], ctc_logits[i])
    samples.append(build_stage1_sample(
        audio_tok, entity_lists[i], batch["text"][i],
        self.tokenizer, embed_layer))

padded = pad_batch(samples)
out = self.lm(**padded)        # HF 會自己用 labels 算 CE loss
ce_loss = out.loss
```

---

## `encode_audio` 的 bf16 寫法

```python
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = self.encoder(input_features)
x = out.last_hidden_state.float()      # 必須轉回 fp32 給可訓練的 adapter

h = self.adapter(x)                    # ⚠️ 必須在 with 外面
ctc_logits = self.ctc_head(h)          # ⚠️ 必須在 with 外面
```

> [!danger] 這兩行絕不能放進 `with`
> | 放錯位置 | 後果 |
> |---|---|
> | 在 **`no_grad` 裡** | adapter / CTC head **拿不到梯度**，永遠學不到。但 loss 仍會降（LoRA 還在學）→ **極難察覺** |
> | 在 **`autocast` 裡** | 可訓練 Linear 用 bf16 算，梯度精度受損 |
>
> `.float()` 放 with 內或外**無功能差異**（autocast 不覆寫明確的型別轉換），放外面只是視覺上更清楚「離開 bf16 區域」。

**為什麼 encoder 用 bf16**：RTX 3090 的 tensor core 支援 bf16 但**不支援 fp32**，實測 759ms → 194ms（**3.92x**）。encoder 凍結所以無梯度精度問題。詳見 [[test.py]]。

---

## 驗證結果（dev-clean, batch=2, RTX 3090）

```bash
python src/model.py
```

| 項目 | 實測 | 預期 | 判定 |
|---|---|---|---|
| 可訓練參數 | **3,888,029** | `2,949,696 + 16,733 + 921,600` | ✅ **精準命中** |
| 有梯度的張量 | **124/124** | adapter 2 + ctc 2 + LoRA 30×2×2 = 120 | ✅ 整條鏈可微分 |
| ctc_loss | 13.39 | S=89→11.9、S=63→17.3，均 14.6 | ✅ |
| ce_loss | **5.60** | ≪ `ln(49152)=10.80` | ✅ **證明預訓練權重確實載入** |

### `3,888,029` 的意義

```
AudioAdapter    2,949,696
CTCHead            16,733
LoRA(asr)         921,600
───────────────────────────
                3,888,029   / 5.7M 預算
```

剩下 1.8M 給 projector（333K）+ LoRA(scoring)（921K）= 1.25M，塞得進去。
**這是整個參數預算推論的最終驗證** —— 架構規模與論文宣稱的 777M 一致。

### `124/124` 的意義

梯度穿過了 `ctc_compress` 的 argmax 選擇、`torch.cat` 的四段拼接、`pad_batch`，到達每一個可訓練張量。

這是最關鍵的驗證。若有張量沒收到梯度，代表那條路斷了（中間用了不可微分操作，或某模組根本沒被 forward 用到）。症狀是「loss 有在降但某個模組完全沒學到」，極難抓。**五秒就能驗。**

```python
out["loss"].backward()
n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
n_tr = sum(1 for p in model.parameters() if p.requires_grad)
print(f"{n_grad}/{n_tr}")
```

### `ce_loss = 5.60` 遠低於 10.80 是好消息

> [!tip] 預訓練模型的初始 CE loss 應明顯低於 ln(vocab)
> `ln(49152) = 10.80` 是「**隨機初始化**的模型」的基準。但 SmolLM2 是**預訓練好**的，它本來就懂英文 —— 即使聽不懂音訊，光靠語言模型能力（加上 prompt 裡的 entity 提示）就能猜出不少下一個字。
>
> 5.60 對應 perplexity ≈ e^5.6 ≈ 270，對 135M 模型在「看不到音訊內容」下預測逐字稿很合理。
>
> **反過來說**：若 `ce_loss ≈ 10.80`，代表 LM 權重根本沒載進來（變成隨機初始化）。這是**權重載入正確性的檢查手法**。

---

## `main()` 做的事

1. 載入 dev-clean 的一個 batch（batch_size=2）
2. 用 [[biasing.py]] 的 `make_stage1_list` 產生每句的 10 個 entity
3. 建模型、印可訓練參數量
4. `forward_stage1` 並印三個 loss，對照理論基準
5. **`backward()` 並確認梯度到達所有可訓練參數**

第 5 步最重要。
