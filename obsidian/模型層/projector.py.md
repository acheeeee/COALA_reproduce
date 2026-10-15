---
tags: [COALA, 模型層, 檔案, Stage2]
層: 模型層
狀態: 完成（2026-10-15）
---

# projector.py

← [[模型層 總覽]] · [[COALA 索引]]

Stage 2 的第一個元件。目前還沒被任何檔案 import（之後由 Stage 2 的模型 / 訓練腳本使用）。

<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：論文 Eq. 1、2 的 discriminative projector —— 把 LM 對每個候選 entity 的 hidden state 變成一個匹配分數 sᵢ。

| 類型 | 內容 |
|---|---|
| import 外部 | `torch`、`torch.nn` |
| import 自己的 | 無 |
| 被誰 import | 尚無（Stage 2 會用） |

**定義的函式 / 類別**

| 名稱 | 一句話 |
|---|---|
| [[DiscriminativeProjector]] | 兩層 MLP + 長度正規化 |
| ↳ [[DiscriminativeProjector.__init__()]] | `Linear(576, D′) → ReLU → Linear(D′, 1)` |
| ↳ [[DiscriminativeProjector.forward()]] | `(N, L, 576)` + mask → `(N,)` |
| [[projector.main()]] | 5 項假資料檢查 |

**定義的常數**：[[D]]、[[D_PRIME]]

**資料流**：
```
h (projector 輸入) (N, L, 576) ──mlp──→ (N, L, 1) ──squeeze──→ f (token 分數) (N, L)
  ──× mask (projector)，sum / count──→ s (entity 分數) (N,)
```
圖中物件：[[h (projector 輸入)]]、[[f (token 分數)]]、[[mask (projector)]]、[[s (entity 分數)]]
<!-- 結構:end -->

---

## 對應論文

$$s_i = \frac{1}{L_i}\sum_{j=1}^{L_i} f(h_{ij}) \qquad f(h_{ij}) = W_2 \cdot \text{ReLU}(W_1 h_{ij} + b_1) + b_2$$

| 符號 | 意義 | 程式 |
|---|---|---|
| h_ij | 第 i 個 entity 第 j 個 token 在 LM 最後一層的 hidden state | `h[i, j]` |
| Lᵢ | 第 i 個 entity 的 token 數 | `mask[i].sum()` |
| D | LM hidden size = 576 | [[D]] |
| D′ | 中間層寬度（論文未給） | [[D_PRIME]] = 576 |

---

## 兩個設計決定

### D′ = 576
```
projector 參數量 = D′(D + 2) + 1
剩餘預算 ≈ 5.7M − 2,413,469（Stage 1, stack=2）− 921,600（LoRA scoring）≈ 2,364,931
→ D′ 上限 ≈ 4091
選 576 → 332,929，預算很寬裕
```

### 為什麼 forward 要吃 mask
121 個候選的 token 數不同，pad 成 `(121, L_max, 576)` 一起算。直接 `.mean(dim=1)` 會錯兩次：
1. 分母變成 L_max 而不是 Lᵢ → 短的 entity 分數被稀釋
2. pad 位置的 f(h) **不是 0**（LM 輸出非零 + bias b₂）→ 分子也被污染

---

## 驗證（2026-10-15，伺服器實測）

```
params: 332,929  (預期 332,929)
s.shape: (121,)  (預期 (121,))
pad 不影響分數: True
與逐個計算一致: True
全部參數有梯度: True
```

| 檢查 | 證明了什麼 |
|---|---|
| params | Eq. 2 的兩層都對、D′ 生效 |
| pad 不影響分數 | mask 確實擋掉 pad |
| 與逐個計算一致 | 分母是 Lᵢ，不是 L |
| 全部參數有梯度 | 整條路可微分 |

---

## 已知的邊界情況（現在不用改）

| 情況 | 後果 | 防法 |
|---|---|---|
| `count = 0` | NaN | `count.clamp(min=1)` |
| pad 位置 h 是 NaN / inf | `NaN × 0 = NaN` | `f.masked_fill(mask == 0, 0)` |

---

## 下一步

`losses.py`：三個 loss 吃的就是這裡輸出的 [[s (entity 分數)]] + E⁺ / E⁻ 遮罩。
