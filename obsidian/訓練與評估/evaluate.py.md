---
tags: [COALA, 評估, 檔案]
層: 訓練與評估
狀態: 能跑，有 2 個 bug
---

# evaluate.py

← [[訓練與評估 總覽]] · [[COALA 索引]]

引用 [[model.py]]、[[dataset.py]]、[[biasing.py]]、[[modules.py]]、[[prompt.py]]。

> [!todo] 狀態（2026-10-16 更新）：程式已寫完、能跑，但有 2 個 bug 待修
> - Bug A：hyp 含換行 → 改 [[transcribe_one()]]
> - Bug B：評估時也過濾了 >30 秒的句子 → 改 [[evaluate.main()]]
> - （舊待辦「generate + decode 未填完」「build_gen_prompt 回傳格式」都已完成：回傳 dict）


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：載入 checkpoint、逐句 greedy 生成轉寫、寫出 hyp 檔給官方 `score.py` 算 WER / U-WER / B-WER。三種模式：不給提示、只給答案、給 10 個 entity。

| 類型 | 內容 |
|---|---|
| import 外部 | `argparse`、`json`（未使用）、`random`、`pathlib`、`torch`、`tqdm`、`transformers` |
| import 自己的 | [[biasing.py]]、[[dataset.py]]、[[model.py]]、[[modules.py]]（`ctc_compress`）、[[prompt.py]]（`build_gen_prompt`） |
| 被誰 import | 無 |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[evaluate.parse_args()]] | → [[args]] |
| [[transcribe_one()]] | 一句 → [[hyp]] |
| [[evaluate.main()]] | 全部句子 → [[hyp TSV]] |

**定義的常數**：[[MAX_NEW_TOKENS]]

**資料流**：
```
input_features → encode_audio → ctc_compress → audio_tok → build_gen_prompt → prompt (生成 prompt)
  → lm.generate → out_ids → decode → hyp → hyp TSV → score.py (官方)
```
圖中物件：[[input_features]]、[[audio_tok]]、[[prompt (生成 prompt)]]、[[out_ids]]、[[hyp]]、[[hyp TSV]]、[[score.py (官方)]]

> [!bug] 目前狀態：能跑，但有 2 個 bug（見 [[evaluate.main()]]）
> - Bug A：hyp 含換行 → [[transcribe_one()]] 改用 `" ".join(text.split())`
> - Bug B：評估也過濾了 >30 秒 → `LibriSpeechDataset(..., max_duration=1e9)`
<!-- 結構:end -->

---

## 功能

產生 ASR 預測並寫成 hyp 檔，供官方 `score.py` 計算 WER / U-WER / B-WER。

常數：`MAX_NEW_TOKENS = 200`

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `parse_args()` | 命令列 | `Namespace` |
| `transcribe_one(model, input_features, entity_list, dev)` | `COALAModel`、`(80,3000)`、`list[str]`、device | `str` — 生成的逐字稿 |
| `main()` | 無 | 無（寫 hyp 檔） |

---

## 這個工具要回答兩個問題

> [!important] 問題 1：模型到底有沒有在用音訊？
> 這是 [[overfit_test.py]] 證明不了的事。
>
> 做法：用 **`--mode nobias`** —— prompt 裡完全不給 entity，只叫它轉寫。
>
> | WER | 結論 |
> |---|---|
> | < 40% | ✅ 真的在聽音訊 |
> | 接近 100% | ❌ 只是在猜，音訊路徑沒被用到 |

**問題 2：數字跟論文差多少？** 見下方對照表。

---

## 命令列參數

| 參數 | 說明 |
|---|---|
| `--ckpt` | **必填** — checkpoint 路徑 |
| `--manifest` | 預設 `data/manifest/dev-clean.tsv` |
| `--out` | **必填** — hyp 檔輸出路徑 |
| `--mode` | `nobias` / `oracle` / `stage1` |
| `--limit` | 只跑前 N 句（快速測試，避免跟訓練搶 GPU） |
| `--seed` | 42 |

### 三種 mode 對應論文 Table 2

| mode | prompt 裡的 entity | 論文對應 |
|---|---|---|
| `nobias` | `[]`（空 → 走 `INSTRUCTION_NO_BIAS`） | **w/o Biasing** |
| `oracle` | 該句真實的 rare words | **Oracle** |
| `stage1` | `make_stage1_list` 的 10 個 | 訓練時的設定 |

---

## 訓練 vs 生成的差異

```
訓練:  [seg1][audio][seg2][逐字稿]  → teacher forcing，算 CE loss
生成:  [seg1][audio][seg2]          → generate() 自己產出
```

評估時沒有答案，所以需要 [[prompt.py]] 的 `build_gen_prompt`（**不含 target**）。

---

## generate 設定

```python
out_ids = model.lm.generate(
    inputs_embeds=prompt["inputs_embeds"],
    attention_mask=prompt["attention_mask"],
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,        # greedy，可重現
    num_beams=1,
    eos_token_id=model.tokenizer.eos_token_id,
    pad_token_id=model.tokenizer.eos_token_id,
)
text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
return text.strip()
```

> [!tip] 走 `inputs_embeds` 的意外好處
> 回傳值**只有新生成的 token**，不含 prompt（因為 prompt 根本沒有對應的 id）→ 直接 `decode` 就是答案。
>
> 用 `input_ids` 時回傳會包含 prompt，要自己切：
> ```python
> new_ids = out_ids[0][inputs["input_ids"].shape[1]:]
> ```
> 這一步很容易算錯（off-by-one、left padding 起點不同），是常見 bug 來源。**這條路自動避開了。**

`do_sample=False` + `num_beams=1` = greedy decoding，結果**完全可重現**。評估一定要可重現。

---

## 載入 checkpoint

```python
ck = torch.load(args.ckpt, map_location=dev)
missing, unexpected = model.load_state_dict(ck["model"], strict=False)
assert len(unexpected) == 0, f"checkpoint 有不認識的鍵: {unexpected}"
model.eval()
model.lm.set_adapter("asr")        # ⚠️ Stage 1 用 asr 那組
```

---

## 輸出格式

官方 `score.py` 要求兩欄 tab 分隔：

```
utt_id<TAB>hypothesis
```

---

## 接官方 score.py

```bash
python /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/score.py \
    --refs /mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/test-clean.biasing_500.tsv \
    --hyps exp/hyp/test-clean.n500.tsv
```

> [!important] 評估指標一律用官方實作
> 自己刻的 WER 在 tokenization、標點、大小寫處理上一定會有微妙差異，**數字就不可比了**。
>
> `score.py` 只用 Python 標準庫（`collections` / `enum` / `argparse` / `logging` / `json`），零外部依賴，直接拿來用。
>
> **你會算 ≠ 你該自己刻。** reproduce 的鐵則是：評估指標用原始作者用的那份實作。

> [!warning] 官方 ref 檔只有 test set
> dev-clean 要另外處理（自己產生 ref 格式，或只看 overall WER）。
> 而且 N=5000 官方沒提供（見 [[biasing.py]]）。

---

## 論文 Table 2 的 COALA 列（對照目標）

格式是 **`B-WER (WER / U-WER)`**，括號裡第一個數字才是整體 WER。

| 設定 | test-clean | test-other |
|---|---|---|
| w/o Biasing | 23.39 (4.54 / 2.34) | 39.49 (8.75 / 5.48) |
| Oracle | 1.77 (1.90 / 1.92) | 6.12 (4.86 / 4.73) |
| N=500 + BTI | 3.25 (2.28 / 2.17) | 9.13 (5.57 / 5.20) |
| N=1000 + BTI | 3.86 (2.50 / 2.34) | 10.59 (5.98 / 5.49) |
| N=5000 + BTI | 6.96 (3.39 / 2.97) | 15.17 (7.33 / 6.50) |
| COALA 不做 BTI, N=5000 | **OOM** | **OOM** |

**三個指標的定義**（論文 3.2）：
- **WER** — 整體
- **U-WER** — unbiased WER，常見字的錯誤率
- **B-WER** — biased WER，**biasing 詞的錯誤率** ← 論文主要關注

---

## 使用建議

```bash
# 先跑 50 句確認 generate 能動，並肉眼看 REF/HYP 對照
python src/evaluate.py --ckpt exp/stage1_tc100/epoch0.pt \
    --out exp/hyp/dev-clean.nobias.tsv --mode nobias --limit 50
```

`main()` 會印出前 3 句的 REF / HYP 對照 —— **這是最直接的判斷**，肉眼看模型轉出來的東西像不像那句話，比任何數字都快。

> [!warning] 評估會跟訓練搶 GPU
> dev-clean 2694 句逐句生成約 **45 分鐘**，會讓訓練變慢。
> 訓練期間只用 `--limit` 跑小樣本，完整評估等訓練結束。

---

## 待處理

- [ ] 填完 `transcribe_one` 的 generate + decode
- [ ] 統一 `build_gen_prompt` 的回傳格式
- [ ] batch 生成（目前 batch_size=1，若太慢需 left padding）
- [ ] dev-clean 的 ref 檔（官方只有 test set）
