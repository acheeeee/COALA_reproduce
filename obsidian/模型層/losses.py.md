---
tags: [COALA, 模型層, 檔案, loss, Stage2]
層: 模型層
狀態: 完成（2026-10-18）
---

# losses.py

← [[模型層 總覽]] · [[COALA 索引]]

**論文的核心貢獻。** 完全不依賴模型：輸入只是「一串分數 + 哪些是正樣本」。目前還沒被 import（之後由 `train_stage2.py` 使用）。

<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：Stage 2 的四個訓練目標 —— baseline 的 disc / Bias-Loss，以及論文提出的 MPD / DPD。

| 類型 | 內容 |
|---|---|
| import 外部 | `torch`、`torch.nn.functional` |
| import 自己的 | 無 |
| 被誰 import | 尚無（`train_stage2.py` 會用） |

**定義的函式**

| 函式 | 論文 | 一句話 |
|---|---|---|
| [[disc_loss()]] | Eq. 3 | 全域 softmax，正樣本互搶 |
| [[mpd_loss()]] | Eq. 4 | 每個正樣本各自對 E⁻ 做 softmax |
| [[dpd_loss()]] | Eq. 5–7 | 每個 entity 各自做二元分類 |
| [[bias_loss()]] | Huang & Wang Eq. 2 | 0.9 × disc + 0.1 × log loss（baseline） |
| [[check()]] | | 測試小工具 |
| [[losses.main()]] | | 圖 2(b) 例子的 5 項測試 |

**輸入物件**：[[s (entity 分數)]]、[[pos (E⁺ 遮罩)]]、[[logp (entity log 機率)]]

**資料流**：[[DiscriminativeProjector]] → [[s (entity 分數)]] + [[pos (E⁺ 遮罩)]] → 四個 loss 之一 → `backward()`
<!-- 結構:end -->

---

## 四個 loss 對照

| loss | 正規化範圍 | 正樣本之間 | 優化的是 | 論文 claim |
|---|---|---|---|---|
| disc | 全部 K 個 | **互搶**（gradient conflict） | 相對排名 | 單獨用會崩潰（實驗 A） |
| Bias-Loss | disc + log loss | 互搶，但 log loss 撐住 | 相對排名 | baseline（實驗 B） |
| **MPD** | 1 個正樣本 + 全部 E⁻ | 獨立 | 相對排名 | 可單獨收斂（實驗 C） |
| **DPD** | 每個 entity 自己 | 獨立 | **絕對**分數 | 可單獨收斂（實驗 D） |

---

## ⭐ 梯度證據（測試 4，論文沒有這張表）

```
            <unb>   Watson  solved  survey  collect
disc:      +0.006  +0.360  −0.384  +0.016  +0.002
mpd:       +0.024  −0.013  −0.085  +0.065  +0.009
dpd:       +0.040  −0.024  −0.135  +0.090  +0.016
```

梯度下降會把**梯度為正**的分數往下推。disc 裡 Watson 是正樣本卻 **+0.36** → 被往下推。MPD、DPD 的兩個正樣本都是負的 → 一起往上推。

**disc 的下界 = ln|E⁺|**：兩個正樣本最好各拿 50% → loss ≥ ln 2 ≈ 0.69；24 個 rare word → ≥ ln 24 ≈ 3.18。也就是說，disc 的最佳解本身就在壓低正樣本 —— 這可以解釋 baseline 為什麼需要輔助 loss。

---

## 數值穩定

| 危險寫法 | 問題 | 這裡的寫法 |
|---|---|---|
| `log(exp(s) / exp(s).sum())` | float32 的 exp 在 x > 88.7 溢位 → nan | `F.log_softmax` |
| `log(exp(a) + exp(b))` | 同上 | `torch.logaddexp`、`torch.logsumexp` |
| `log(1 − sigmoid(x))` | σ ≈ 1 時 log(0) | `F.logsigmoid(-x)` |

測試 3（s = ±300）：100.0 / 0.0 / 0.0，全部有限值 ✅

---

## log loss 的來源（疑點 #10）

- COALA 論文只說 baseline「requires an auxiliary log loss (weight 0.1)」，沒給公式
- **官方 code（`Guo0911/COALA`）只有 MPD、DPD，沒有 baseline 的實作**
- 定義出自 [Huang & Wang 2025](https://arxiv.org/html/2510.23849v1) Eq. 2：`(1−β)·L_log + β·L_disc`，β = 0.9
- L_log = 正樣本被 LM **生成出來**的負 log 機率（lm_head 詞彙空間），不是 BCE

**自選設定**：log loss 對正樣本**加總**、**排除 `<unbiased>`**（照原論文）。

和原論文的差別：原版 disc 也是加總，COALA Eq. 3 改成平均；這裡照 COALA。

---

## 和官方 code 對照（2026-10-18）

| | 官方 | 我的 | 一致？ |
|---|---|---|---|
| MPD | for 迴圈逐個正樣本 `logsumexp(stack([pos, neg_lse]))` | `logaddexp` 一次算完 | ✅ 數學相同 |
| DPD | `binary_cross_entropy_with_logits` × 2，各自平均 | `logsigmoid` × 2，各自平均 | ✅ 數學相同 |
| disc / Bias-Loss | **沒有** | 有 | — |
