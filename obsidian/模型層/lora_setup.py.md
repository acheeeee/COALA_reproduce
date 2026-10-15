---
tags: [COALA, 模型層, 檔案]
層: 模型層
狀態: 完成
---

# lora_setup.py

← [[模型層 總覽]] · [[COALA 索引]]

被 [[model.py]] 引用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：LoRA 設定工廠（r=16，掛 q_proj / v_proj），以及切換、凍結、診斷兩組具名 adapter（asr / scoring）的工具。

| 類型 | 內容 |
|---|---|
| import 外部 | `torch`、`peft`（`LoraConfig`、`get_peft_model`）、`transformers` |
| import 自己的 | 無 |
| 被誰 import | [[model.py]]（`LM_ID`、`make_lora_config`） |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[make_lora_config()]] | → [[lora_config]] |
| [[count_trainable()]] | 數可訓練參數 |
| [[set_active()]] | 啟用一組或多組 adapter |
| [[freeze_adapter()]] | 凍結某一組 |
| [[which_trainable()]] | 診斷：requires_grad 落在哪組 |
| [[which_gets_grad()]] | 診斷：backward 後誰有梯度 |
| [[lora_setup.main()]] | 兩組共存、切換、Stage 2 情境測試 |

**定義的常數**：[[LM_ID]]、[[LORA_RANK]]、[[LORA_ALPHA]]、[[LORA_DROPOUT]]、[[TARGET_MODULES]]

**之後會用到的**：[[set_active()]] + [[freeze_adapter()]] 就是 Stage 2「兩組都啟用、只訓練 scoring」的做法。
<!-- 結構:end -->

---

## 功能

LoRA 設定工廠。論文未給任何 LoRA 設定（TODO 疑點 #3），**rank 由參數預算反推**。

---

## 設定

```python
LM_ID          = "HuggingFaceTB/SmolLM2-135M-Instruct"
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
bias           = "none"          # 不訓練原本的 bias
task_type      = "CAUSAL_LM"
```

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `make_lora_config()` | 無 | `peft.LoraConfig` |
| `count_trainable(model)` | `nn.Module` | `int` |
| `main()` | 無 | 無（驗證兩組 adapter 共存 + forward） |

---

## ⭐ rank=16 是怎麼推出來的

> [!important] 用論文有給的數字反推它沒給的數字
> 論文給了總參數量 **777M**，而實測 encoder 636.8M + LM 134.5M = 771.3M
> → **可訓練預算 = 5.7M**（見 [[check_models.py]]）

```
5.7M − adapter/CTC 2.97M − projector 0.33M ≈ 2.4M  → 兩組 LoRA 各 1.2M

SmolLM2-135M 架構：30 層, hidden 576, 9 heads (head_dim 64), 3 KV heads (GQA)
  q_proj 576→576:  r×576 + 576×r = 18,432     (r=16)
  v_proj 576→192:  r×576 + 192×r = 12,288
  每層 30,720  ×  30 層  =  921,600
```

**實測 `trainable params: 921,600` — 與推算完全一致** ✅

落在 1.2M 預算內。若訓練不收斂，這是第一個可調的旋鈕，而且**知道合理範圍在哪**。

---

## 層名稱確認

```bash
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M-Instruct'); print([n for n,_ in m.named_modules() if 'proj' in n][:12])"
```

```
model.layers.0.self_attn.q_proj / k_proj / v_proj / o_proj
model.layers.0.mlp.gate_proj / up_proj / down_proj
...
```

確認 `q_proj` / `v_proj` 存在（有些模型叫 `query`、`Wq`）。

---

## 兩組具名 adapter

論文圖 2(a)：Backbone LM 內部並排兩組 LoRA，標著 **ASR** 和 **Scoring**。

```python
model = get_peft_model(lm, make_lora_config(), adapter_name="asr")
model.add_adapter("scoring", make_lora_config())
model.set_adapter("asr")        # 切換；另一組自動凍結
```

實測行為：

```
134,515,008   原始 LM
135,436,608   + asr        (+921,600)
136,358,208   + scoring    (+921,600)
可訓練始終 921,600          ← set_adapter 只啟用一組
```

> [!success] 正好符合論文 Stage 2 的需求
> Stage 1 訓練 `asr`；Stage 2 「All previously trained parameters are frozen」→ 凍結 `asr`、只訓練新的 `scoring` + projector。
> 圖 2(a) 裡 `ASR` 那組標 ❄️、`Scoring` 那組標 🔥，正是這個機制。

---

## ⚠️ 凍結與包 peft 的順序不能反

> [!danger] 這是會讓 loss 完全不降、且不報錯的 bug
> `get_peft_model` **本身就會**自動凍結所有非 LoRA 參數。
>
> ```python
> # ✓ 正確
> for p in lm.parameters():
>     p.requires_grad = False
> model = get_peft_model(lm, cfg, adapter_name="asr")
>
> # ✗ 錯誤：把剛建好的 LoRA 一起凍結 → 可訓練參數變 0
> model = get_peft_model(lm, cfg, adapter_name="asr")
> for p in model.parameters():
>     p.requires_grad = False
> ```
>
> 先手動凍結其實是多餘的（但無害，而且意圖明確）。真正的風險是順序顛倒。

---

## 兩個設定選擇的依據

> [!info] 為什麼只選 `q_proj` + `v_proj`，不選全部的 q/k/v/o
> LoRA 原論文（Hu et al. 2021）**Table 5** 的消融實驗：在**固定參數預算**下比較「只調 Wq」、「Wq+Wk」、「Wq+Wv」、「全部四個」，結論是 **Wq + Wv 最好**。
>
> 直覺上 query 決定「要找什麼」、value 決定「取回什麼內容」，這兩個是注意力機制中語意最關鍵的部分；key 和 output projection 的邊際效益較低。

> [!info] `alpha / r = 2` 代表什麼
> LoRA 的更新量是 `ΔW = (alpha / r) × B @ A`。那個比值是**縮放係數**，效果類似對 LoRA 分支單獨設一個學習率倍率。
>
> 好處是：**改 r 時不用重調學習率**。把 r 從 16 改成 32 時，只要 alpha 也變 64（維持比值 2），更新量尺度不變，原本的 lr 還能用。
>
> 比值 2 是常見預設（也有人用 1 或 4）。論文沒給，這是自選的，要記錄。

---

## 驗證

```bash
python src/lora_setup.py
```

```
total parameter: 134.5M
trainable params: 921,600 || all params: 135,436,608 || trainable%: 0.6805
trainable params: 921,600 || all params: 136,358,208 || trainable%: 0.6759
ASR:     trainable params: 921,600
Scoring: trainable params: 921,600
torch.Size([1, 20, 49152])
```

> [!success] 最後那行是關鍵
> `(1, 20, 49152)` 代表 LM 吃進**自己造的 576 維向量**（不是真的 token id），正常吐出詞彙分布。
> **這證明 `inputs_embeds` 這條路通了** —— audio token 真的能當 prompt 用（見 [[prompt.py]]）。

---

## 踩過的坑：transformers 5.x 的 dtype

> [!danger] `RuntimeError: expected m1 and m2 to have the same dtype, but got: float != c10::BFloat16`
> **v4**：`from_pretrained` 一律轉成 fp32
> **v5**：沿用 checkpoint 自己的 dtype → SmolLM2 載成 **bf16**、Whisper 載成 **fp32**
>
> 這會在完整 pipeline 再撞一次：Whisper(fp32) → AudioAdapter(fp32) → SmolLM2(**bf16**) ✗
>
> **決定：全專案統一 fp32**（VRAM 還剩 21GB，不缺；fp32 無數值風險）
>
> ```python
> AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
> ```
>
> **通則：所有 `from_pretrained` 都明確指定 dtype，不依賴預設。**
> （transformers 5.x 參數名是 `dtype`，v4 是 `torch_dtype`）
