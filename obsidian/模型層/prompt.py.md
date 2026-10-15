---
tags: [COALA, 模型層, 檔案]
層: 模型層
狀態: 完成
---

# prompt.py

← [[模型層 總覽]] · [[COALA 索引]]

被 [[model.py]] 和 [[evaluate.py]] 引用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：把「文字 + 音訊向量」組成 LM 的輸入序列 —— 訓練版（含答案和 labels 遮罩）、生成版（只到 assistant 開頭），以及 batch padding。

| 類型 | 內容 |
|---|---|
| import 外部 | `torch`（main 裡另外 import `transformers`） |
| import 自己的 | 無 |
| 被誰 import | [[model.py]]（`build_stage1_sample`、`pad_batch`）、[[evaluate.py]]（`build_gen_prompt`） |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[build_seg1()]] | → [[seg1]] 文字 |
| [[build_seg2()]] | → [[seg2]] 文字（含 entity） |
| [[encode()]] | 文字 → id tensor |
| [[build_stage1_sample()]] | 一句 → [[sample (stage1 序列)]] |
| [[pad_batch()]] | 多句 → [[padded]] |
| [[build_gen_prompt()]] | 一句 → [[prompt (生成 prompt)]] |
| [[prompt.main()]] | shape 與 labels 檢查 |

**定義的常數**：[[IGNORE_INDEX]]、[[SYSTEM_PROMPT]]、[[INSTRUCTION_TEMPLATE]]、[[INSTRUCTION_NO_BIAS]]

**資料流**：
```
seg1 / seg2 / target → encode → *_ids → embed_layer → *_emb
cat([seg1_emb, audio_tok, seg2_emb, target_emb]) → inputs_embeds + labels + attention_mask
```
圖中物件：[[seg1]]、[[seg2]]、[[target]]、[[embed_layer]]、[[audio_tok]]、[[inputs_embeds]]、[[labels]]、[[attention_mask]]
<!-- 結構:end -->

---

## 功能

組裝 LM 的輸入序列。論文未給完整 instruction（TODO 疑點 #7），以下是**自行設計**。

---

## 核心技術點：為什麼要走 `inputs_embeds`

audio token 是 576 維向量，**沒有對應的 token id**，不在詞彙表裡。所以不能用平常的 `lm(input_ids=...)`：

```
lm(input_ids=...)       # 吃 token id，模型內部自己查 embedding 表
lm(inputs_embeds=...)   # 直接吃 576 維向量，跳過查表  ← 要走這條
```

組 prompt 的流程變成：

```
文字部分  →  tokenizer  →  token ids  →  lm.get_input_embeddings()(ids)  →  (n, 576)
音訊部分  →  已經是 (L, 576)
                              ↓ torch.cat
              (L + n, 576) → lm(inputs_embeds=...)
```

**你要自己做 embedding 查表這一步**，因為只有這樣才能把音訊向量插進序列中間。

> [!info] 為什麼這招可行
> LM 的 Transformer 層其實**不知道「token」這回事**，它只看到一串 576 維向量。token id 和查表只是產生那些向量的手段。
> 所以可以塞任何 576 維向量進去 —— 這就是 SLM（speech-augmented LM）的核心把戲。
> 論文圖 (a) 的 **Audio Tokens** 就是這個。它們佔據 token 的位置，但不對應詞彙表裡任何詞。

---

## 第二個核心技術點：labels 要遮罩

Stage 1 要訓練「聽音訊 → 輸出逐字稿」。但 loss **只能算在逐字稿那幾個位置** —— 不希望模型去「預測 instruction 的下一個字」，那是你給它的輸入。

HuggingFace 的慣例是用 **`-100`** 標記「這個位置不算 loss」：

```
序列:    [seg1][audio][seg2][transcription]
labels:  [-100][-100 ][-100][ 真實 token ids ]
                                ↑ 只有這段算 CE loss
```

模型內部看到 `-100` 就跳過。這是 `nn.CrossEntropyLoss(ignore_index=-100)` 的預設行為。

---

## 序列結構（訓練，四段）

```
seg1   = <|im_start|>system\nYou are a speech recognition assistant.<|im_end|>\n<|im_start|>user\n
audio  = ctc_compress 輸出的 (L, 576) 向量
seg2   = Transcribe the speech. The following entities may appear: {10 個 entity}.<|im_end|>\n<|im_start|>assistant\n
target = {逐字稿}<|im_end|>
```

**音訊放在 user turn 的開頭**，對應論文 Figure 2(a) 的 `[Audio Tokens][Instruction Embeds][Entity Embeds]` 順序。

### SmolLM2 的 chat template

```python
'<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHELLO<|im_end|>\n<|im_start|>assistant\n'
bos: <|im_start|>   eos: <|im_end|>   pad: <|im_end|>
```

把預設系統提示（17 token，對 ASR 毫無幫助）換成任務相關的短句。

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `build_seg1(tokenizer)` | tokenizer | `str` |
| `build_seg2(entity_list, tokenizer)` | `list[str]`（**空的話走 no-bias 分支**）、tokenizer | `str` |
| `encode(text, tokenizer, device)` | `str`、tokenizer、device | `(n,)` long tensor，`add_special_tokens=False` |
| `build_stage1_sample(audio_tokens, entity_list, target_text, tokenizer, embed_layer)` | `(L,576)`、`list[str]`、`str`、tokenizer、`lm.get_input_embeddings()` | `dict` ↓ |
| `pad_batch(samples, pad_side="right")` | `list[dict]`、`str` | `dict` ↓ |
| `build_gen_prompt(audio_tokens, entity_list, tokenizer, embed_layer)` | 同上但無 target | `dict` ↓ |

常數：

```python
IGNORE_INDEX         = -100
SYSTEM_PROMPT        = "You are a speech recognition assistant."
INSTRUCTION_TEMPLATE = "Transcribe the speech. The following entities may appear: {entities}."
INSTRUCTION_NO_BIAS  = "Transcribe the speech."          # 對應論文 w/o Biasing
```

### `build_stage1_sample` 回傳（單句，無 batch 維）

| 鍵 | shape |
|---|---|
| `inputs_embeds` | `(n1+L+n2+m, 576)` |
| `labels` | `(n1+L+n2+m,)` — 前 `n1+L+n2` 全是 `-100` |
| `attention_mask` | `(n1+L+n2+m,)` — 全 1 |

### `pad_batch` 回傳（加上 batch 維）

`(B, max_tot, 576)` / `(B, max_tot)` / `(B, max_tot)`

⚠️ `labels` 的 padding 填 **`IGNORE_INDEX`** 而非 0 —— padding 位置不能算 loss。

### `build_gen_prompt` 回傳

**回傳 dict**（與 [[evaluate.py]] 的用法一致）：

```python
return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
```

| 鍵 | shape |
|---|---|
| `inputs_embeds` | `(1, n1+L+n2, 576)` |
| `attention_mask` | `(1, n1+L+n2)` |

**不含 target** —— 評估時沒有答案，要讓 `generate()` 自己產出。

```
訓練:  [seg1][audio][seg2][逐字稿]  → teacher forcing，算 CE loss
生成:  [seg1][audio][seg2]          → generate() 自己產出
```

⚠️ 注意 `.unsqueeze(0)` 加 batch 維之後，**序列長度在 `shape[1]` 不是 `shape[0]`**：

```python
inputs_embeds = torch.cat([seg1_emb, audio_tokens, seg2_emb], dim=0).unsqueeze(0)
total = inputs_embeds.shape[1]            # ← 不是 shape[0]（那是 batch=1）
attention_mask = torch.ones(1, total, dtype=torch.long, device=dev)
```

> [!tip] 用 dict 而非 tuple 的好處
> 之後要多回傳東西（例如 KV cache 的 `past_key_values`）時不用改呼叫端。

---

## Token 預算（實測）

```
文字 59 + audio 89 + target 12 = 160 / 2048（用掉 8%）
```

> [!success] Stage 2 的 KV cache 空間
> Stage 2 的結構是 `seg1(16) + audio(89) + instruction + 單一 entity ≈ 128`，
> 其中 **`seg1 + audio` 這 105 個 token 對 121 次 forward 完全相同**
> → KV cache 可省 **82%** 重複計算。這是 Stage 2 能跑得完的關鍵。

---

## 三個注意事項

> [!note] 為什麼 target 要加 `eos_token`
> 不加的話模型推論時**不知道何時該停**，會一直生成到 context 滿。加了才會學到「說完逐字稿就輸出 `<|im_end|>`」。

> [!warning] padding 方向
> | 情境 | 方向 | 理由 |
> |---|---|---|
> | **訓練** | 補右邊 | `labels` 用 `-100`、`attention_mask` 用 0 即可，teacher forcing 不受影響 |
> | **生成** | **必須補左邊** | causal LM 接著序列**最後一個位置**往下生成，補右邊會從 padding 位置開始接續 → 產出垃圾 |
>
> ```
> 補右邊 ✗  [真實內容][PAD][PAD]
>                          ↑ 從這裡接續 → 垃圾
> 補左邊 ✓  [PAD][PAD][真實內容]
>                           ↑ 從真實內容結尾接續
> ```
> HuggingFace 的慣例是生成時設 `tokenizer.padding_side = "left"`。
> 或推論時乾脆 `batch_size=1` 就不用處理（Stage 2 本來就是 batch 1）。

> [!danger] 踩過的坑：`len()` 算字元數不是 token 數
> ```python
> m = len(target)              # ✗ target 是字串 → 61 個「字元」
> m = target_ids.shape[0]      # ✓ 12 個「token」
> ```
> 症狀：`attention_mask` 是 209 而 `inputs_embeds` 是 160。
> `labels` 剛好對是因為直接用了 `target_ids` 沒經過 `m`。
>
> **長度一律用 `.shape[0]`。**

---

## 驗證

```bash
python src/prompt.py
```

```
inputs_embeds  (160, 576)
labels         (160,)
attention_mask (160,)

算 loss 的位置: 12  (target token 數 = 12)
文字 token 總數: 59
```

**判準**：
- 三個 shape 的第一維**必須完全相同**（`59 + 89 + 12 = 160`）
- `算 loss 的位置` 必須等於 `target token 數`
- 文字 token 總數在 60~80 之間
