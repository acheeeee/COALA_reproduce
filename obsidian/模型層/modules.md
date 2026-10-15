---
tags: [COALA, 模型層, 檔案]
層: 模型層
狀態: 完成
---

# modules.py

← [[模型層 總覽]] · [[COALA 索引]]

被 [[model]] 引用。

---

## 功能

audio adapter、CTC head、CTC 壓縮、CTC loss。

---

## `AudioAdapter(nn.Module)`

把 Whisper encoder 輸出接到 LM 的 embedding 空間。

```
__init__(d_in=1280, d_out=576, stack=4)
forward(x)   (B, 1500, 1280) → (B, 375, 576)
```

**做法：frame stacking** — 把相鄰 4 個 frame 的向量**直接接起來**（`reshape` 成 `(B, T//4, 1280*4)`），再用 `Linear(5120, 576)` 壓到 576。

```python
def __init__(self, d_in=1280, d_out=576, stack=4):
    super().__init__()                           # 呼叫父類別 nn.Module 的 init
    self.stack = stack
    self.proj = nn.Linear(d_in * stack, d_out)   # 5120 → 576

def forward(self, x):
    B, T, d_in = x.shape
    x = x.reshape(B, T // self.stack, d_in * self.stack)
    return self.proj(x)
```

參數量：`5120 × 576 + 576 = 2,949,696`

### 為什麼選 frame stacking

| 做法 | 優點 | 缺點 |
|---|---|---|
| **frame stacking** ✅ | **無資訊損失**（只是重新排列） | 維度膨脹，Linear 參數隨 k 成正比 |
| strided conv | 可學習、平滑融合鄰近 frame | 降採樣率只能是 2 的次方，壓到 40 要疊 5 層 |
| pooling | 零參數 | **直接丟掉資訊** |

真正的大幅壓縮交給 CTC，因為它知道「哪些 frame 是 blank」，是**內容相關**的壓縮，比固定倍率聰明得多。

> [!info] reshape 為什麼能正確接相鄰 frame
> tensor 記憶體是 row-major（最後一維連續）。`(B,T,D)` 中 `(b,t,d)` 的位移是 `b·TD + t·D + d`；reshape 成 `(B,T/4,4D)` 後 `(b,t',j)` 的位移是 `b·TD + t'·4D + j`。
> `t'=0` 時 `j` 掃過 `0..4D-1`，剛好涵蓋 `t=0,1,2,3` 的全部 `d` → **相鄰 4 個 frame 被接到同一列** ✓
>
> ⚠️ 若寫成 `x.transpose(1,2).reshape(...)` 也可能得到同樣的 shape，但接到一起的會是跨越整段音訊的數值 → adapter 學不到東西而且你不會發現。**shape 對了不代表內容對了。**

---

## `CTCHead(nn.Module)`

```
__init__(d_in=576, vocab_size=29)
forward(h)   (B, 375, 576) → (B, 375, 29)
```

參數量：`576 × 29 + 29 = 16,733`

> [!important] 為什麼詞彙是 29 而不是 LM 的 49152
> **參數預算直接把答案逼出來**（論文沒寫，TODO 疑點 #6）：
>
> | CTC 詞彙 | 參數量 | 佔 5.7M 預算 |
> |---|---|---|
> | BPE 49152 | `576×49152+49152` = **28.3M** | **497%** ❌ 爆 5 倍 |
> | 字元 29 | **18.4K** | 0.3% ✅ |
>
> 附帶好處：字元級對罕見字友善。BPE 遇到 `craswellers` 會切得亂七八糟，字元級永遠拼得出來。對一篇做 rare word biasing 的論文是合理選擇。
>
> 詞彙內容已用實際資料確認（見 [[dataset]]）：空格 + 撇號 + a-z = 28，加 blank = 29。

---

## `ctc_compress(hidden, logits, blank_id=0)`

| 參數 | shape |
|---|---|
| `hidden` | `(L, 576)` — adapter 輸出 |
| `logits` | `(L, 29)` — CTC head 輸出 |
| **回傳** | `(L', 576)`，`L' << L` |

**邏輯**（CTC greedy decoding 的標準語意）：保留「預測不是 blank」**且**「與前一個 frame 預測不同」的位置。

```python
pred      = logits.argmax(dim=-1)
not_blank = pred != blank_id
change    = pred[1:] != pred[:-1]
first     = torch.ones(1, dtype=torch.bool, device=pred.device)   # ⚠️ device！
changed   = torch.cat([first, change])
keep      = not_blank & changed
return hidden[keep]
```

驗算 `pred = [a, a, blank, a]`（真實是 "aa"，blank 用來分隔兩個相同字母）：

```
not_blank = [T, T, F, T]
changed   = [T, F, T, T]
keep      = [T, F, F, T]   → 保留 2 個 token ✓
```

有 blank 隔開的兩個 `a` 算兩個，沒隔開的算一個 —— 這就是 blank 存在的意義。

### 為什麼需要壓縮

375 個 audio token 對 SmolLM2 的 **2048** context window 來說太多，而 **Stage 2 要對一個 utterance 跑 121 次 forward** → 這就是論文 Table 2 N=5000 OOM 的機制。

一句話平均只有 40 個字，所以 375 個 frame 裡絕大多數是 blank 或重複。

### 檢查指標

> [!tip] 訓練後壓縮結果應接近**實際字元數**
> 第一句的 `ctc_ids` 是 89 → 理想狀態是 `375 → 89`。
> 隨機初始化時是 **158**（方向對但還沒學會）。
>
> 這個數字是判斷「CTC 是否真的學到對齊」的直接證據。

### 兩個坑

> [!danger] `device` 不能省
> `torch.tensor([True])` 會建在 **CPU**。CPU 測試時完全正常，上 GPU 後 `torch.cat([cpu, cuda])` 直接報錯：
> ```
> RuntimeError: Expected all tensors to be on the same device
> ```
> **通則：函式內部新建 tensor 一律寫 `device=某個輸入.device`。**

> [!warning] 未處理的邊界情況
> 若某句 CTC 預測**全是 blank**，`keep` 全為 False → 回傳 `(0, 576)` 空 tensor。
> 訓練初期模型還沒學會時很可能發生，之後要處理（至少保留 1 個 token）。

> [!note] 梯度可以穿過 argmax
> argmax 本身不可微分，但它只用來「挑選哪些位置」，梯度沿著**被選中的 `hidden`** 流回去。選擇不需要梯度，被選中的內容需要。這在 PyTorch 是合法且常見的做法。
>
> [[model]] 的 `124/124 有梯度` 驗證了這一點。

---

## `compute_ctc_loss(logits, targets, target_lengths, blank_id=0)`

| 參數 | shape / 說明 |
|---|---|
| `logits` | `(B, T, C)` — CTC head 輸出 |
| `targets` | `(B, S)` — padded 字元 id（`collate_fn` 的 `ctc_ids`） |
| `target_lengths` | `(B,)` — 真實字元數（`collate_fn` 的 `ctc_lens`） |
| **回傳** | 純量 |

```python
log_probs = logits.log_softmax(dim=-1).transpose(0, 1)      # → (T, B, C)
B, T = logits.shape[0], logits.shape[1]
input_lengths = torch.full((B,), T, dtype=torch.long, device=logits.device)
return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                  blank=blank_id, reduction="mean", zero_infinity=True)
```

> [!danger] 四個介面陷阱（公認的 bug 熱點）
> **1. 要 `(T, B, C)` 維度順序**，與 PyTorch 慣例 `(batch, seq, feature)` 相反 → 必須 `transpose(0, 1)`
>
> **2. 吃 log 機率不是 logits** → 要先 `log_softmax(dim=-1)`。忘記**不會報錯**，只會算出一個看起來合理但錯誤的數字。最陰險的一個
>
> **3. `target_lengths` 傳錯不會報錯** → CTC 只要求 `target_length ≤ S`，填 172 它就認真把 padding 的 0（= blank id）當成真實目標，於是模型被訓練成「要在這裡輸出 blank」。典型靜默失敗
>
> **4. `zero_infinity=True`** → 若某句字元數比時間步多（CTC 數學上無解）會得到 `inf`，一個 inf 就毀掉整個 batch 的梯度。本專案 375 步 vs 最多 172 字元不會發生，但這是標準防護

`input_lengths` 全部等於 T，因為 Whisper 固定 30 秒、時間軸沒有 padding。

### 初始 loss 的理論基準（判斷有沒有 bug 的錨點）

```
loss ≈ (T·ln(C) − ln(對齊路徑數)) / S
```

T=375, C=29，用 Stirling 近似：

| S（目標字元數） | 預期初始 loss |
|---|---|
| 63 | 17.3 |
| 89 | 11.9 |
| 133 | 7.7 |
| 172 | 5.8 |

實測 batch(89,63,172,133) → **10.18**，四者平均 10.7，吻合 ✓

> [!warning] `ln(C) = 3.37` 不是 CTC 的基準
> 那是**每個 frame** 的 cross-entropy。CTC 用 `reduction="mean"` 是除以**目標字元數**，所以初始值在 10~17。**句子越短 loss 越高。**
>
> 訓練時要觀察的是「從 10 附近開始下降」。卡在 10 不動 → CTC 沒在學。

### `log_softmax` 的正確理解

```python
softmax([1,2,3])     = [e¹,e²,e³] / (e¹+e²+e³) = [0.090, 0.245, 0.665]
log_softmax([1,2,3]) = [-2.407, -1.407, -0.407]       ← 全部是負數
```

> [!danger] 我註解裡曾寫錯的地方
> ```python
> # ✗ 錯的：X=[1,2,3] -> X.log_softmax() = [1/6, 2/6, 3/6]
> ```
> 那是「**除以總和**」，不是 softmax。softmax 要**先取 exp 再 normalize**。
>
> 取 exp 的作用是讓大的更大（指數放大差距），這也是它叫 **soft**max 的原因 —— 是 max 的平滑版本。

**檢查手法**：log 機率永遠是負的。印出 `log_probs` 看到正數 → 忘記做 log_softmax。

---

## 驗證

```bash
python src/modules.py
```

```
adapter: (2, 1500, 1280) -> (2, 375, 576)
ctc    : (2, 375, 576) -> (2, 375, 29)

adapter 參數量: 2,949,696
ctc 參數量    : 16,733
合計          : 2,966,429  (預算 5,700,000)

--- CTC loss ---
loss = 10.1801
```

剩餘預算：`5.7M − 2.97M = 2.73M`，再扣 projector（約 333K）→ 約 **2.4M 給兩組 LoRA**，每組 1.2M → 推出 LoRA rank ≈ 16（見 [[lora_setup]]）。
