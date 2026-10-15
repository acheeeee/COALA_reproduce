---
tags: [COALA, 工具, 檔案, 驗證]
層: 工具
狀態: 完成
---

# check_models.py

← [[工具 總覽]] · [[COALA 索引]]

**一次性執行。** 不被任何檔案引用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：量兩個預訓練模型的參數量與維度 → 算出 **5.7M 可訓練預算**。

| 類型 | 內容 |
|---|---|
| import 外部 | `transformers`（`WhisperModel`、`AutoModelForCausalLM`） |
| import 自己的 | 無 |
| 被誰 import | 無 |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[count_params()]] | 數全部參數 |
| [[human()]] | 數字 → `"636.8M"` |
| [[check_models.main()]] | 量測 + 印出預算 |

**量到的關鍵值**：[[D]] = 576、encoder 636.8M、LM 134.5M
<!-- 結構:end -->

---

## 功能

驗證論文 3.4 節的參數量說法，並量出 `D`（projector 的輸入維度）。

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `count_params(module)` | `nn.Module` | `int` — `sum(p.numel() for p in module.parameters())` |
| `human(n)` | `int` | `str` — 例 `"640.1M"` |
| `main()` | 無 | 無（印規格表） |

```bash
HF_HOME=/mnt/disk2/M11515125/hf_cache python src/check_models.py
```

第一次跑會下載 Whisper 的 6.17GB。

---

## 實測結果

| 項目 | 實測 | 論文 | 備註 |
|---|---|---|---|
| Whisper-large-v2 **encoder** | **636.8M** | 640M | ✅ 吻合（論文四捨五入） |
| 整個 `WhisperModel`（含 decoder） | 1543.3M | — | ⚠️ decoder 白佔 906M |
| Whisper encoder `d_model` | **1280** | 未寫 | → adapter 的**輸入**維度 |
| Whisper encoder 層數 | 32 | 未寫 | |
| SmolLM2-135M-Instruct | **134.5M** | 135M | ✅ 吻合 |
| SmolLM2 `hidden_size` | **576** | 未寫 | ⭐ **= Eq. 2 的 `D`** |
| SmolLM2 `vocab_size` | 49152 | 未寫 | |
| SmolLM2 層數 | 30 | 未寫 | → LoRA 掛在這 30 層 |
| encoder + LM | 771.3M | 777M | **差額 5.7M** |

---

## ⭐ 5.7M 這個數字是整個實作的關鍵

> [!important] 它是所有可訓練元件的總預算
> 論文沒給 adapter 架構、CTC 詞彙、projector 的 D′、LoRA rank（TODO 疑點 #3~#6），
> 但給了 **777M 總參數量** → **可以用它反推那些沒寫的設定**。
>
> **reproduce 常用技巧：用論文有給的數字去夾擠它沒給的數字。**

### 已靠它定案兩件事

**1. CTC 詞彙必須是字元級（29），不可能是 BPE**

| CTC 詞彙 | 參數量 | 佔 5.7M |
|---|---|---|
| BPE 49152 | `576×49152+49152` = 28.3M | **497%** ❌ |
| 字元 29 | 18.4K | 0.3% ✅ |

**2. LoRA rank ≈ 16**（見 [[lora_setup.py]]）

```
5.7M − adapter/CTC 2.97M − projector 0.33M ≈ 2.4M → 兩組各 1.2M
30 層 × (q_proj 18,432 + v_proj 12,288) = 921,600  (r=16) ✅
```

### 最終驗證

```
adapter 2,949,696 + CTC 16,733 + LoRA(asr) 921,600 = 3,888,029   ← 實測吻合
再加 Stage 2 的 projector 333K + LoRA(scoring) 921K = 5.14M       ← 塞得進 5.7M ✅
```

---

## 順便發現的浪費

`WhisperModel` 總共 1543.3M，減去 encoder 636.8M → **decoder 白佔 906M**，而且下載的 6.17GB 有一大半用不到。

目前在 CPU 上只是浪費 RAM。[[model.py]] 裡用 `.get_encoder()` 取出 encoder 後，decoder 會在 `__init__` 結束後被垃圾回收，所以**上 GPU 的只有 encoder 的 2.43 GB**（見 [[test.py]] 的 VRAM 實測）。

> [!tip] 更乾淨的做法
> 可以用 `WhisperForConditionalGeneration` 之類的方式只載 encoder，或在載入後明確 `del whisper.decoder`。目前不影響，記著就好。
