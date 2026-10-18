---
tags: [COALA, 函式, 模型層, loss, Stage2]
種類: 函式
檔案: losses.py
---
# bias_loss()

← [[losses.py]] · [[函式 總覽]]

**功能**：baseline 的 Bias-Loss = disc loss + 輔助 log loss（Huang & Wang 2025 Eq. 2），論文實驗 B 的對照組。

$$\mathcal{L}_{bias} = (1-\beta)\,\mathcal{L}_{log} + \beta\,\mathcal{L}_{disc}, \quad \mathcal{L}_{log} = -\sum_{i=1}^{M} l_i \log P(e_i \mid X), \quad \beta = 0.9$$

```python
def bias_loss(s, pos, logp, beta=0.9):
    pos_log = pos.clone()
    pos_log[0] = False                     # 排除 <unbiased>
    log_loss = -logp[pos_log].sum()        # 負號 + 加總 → 純量；空集合 = 0
    loss = (1 - beta) * log_loss + beta * disc_loss(s, pos)
    return loss
```

## 輸入 → 輸出

| 參數 | shape | 物件 |
|---|---|---|
| `s` | `(K,)` | [[s (entity 分數)]] |
| `pos` | `(K,)` bool | [[pos (E⁺ 遮罩)]] |
| `logp` | `(K,)` | [[logp (entity log 機率)]]（8b 才會算出來） |
| `beta` | `float` | 0.9 → log loss 權重 0.1 |
| **輸出** | `()` | 純量 loss |

## 內部物件

| 名稱 | 內容 |
|---|---|
| `pos_log` | `pos` 的**複本**，index 0 改成 False |
| `log_loss` | `−Σ logp` over 正樣本（不含 `<unbiased>`） |

## 三個設計決定（自選，要記錄）

| 決定 | 理由 |
|---|---|
| **排除 index 0** | 原論文加總從 i = 1 開始。`<unbiased>` 不是真的詞，教 LM「生成 `<unbiased>` 這串字」沒意義 |
| **加總，不平均** | 照原論文；空集合（句子沒有 rare word）自然 = 0。平均的話空集合會變 NaN |
| **`clone()`** | 直接改 `pos[0]` 會改到呼叫端的遮罩，`disc_loss(s, pos)` 就拿到被改過的版本 |

## log loss 到底是什麼
**不是 BCE**，而是 LM 透過 lm_head（49152 詞彙）把正樣本 entity **一個 token 一個 token 生成出來**的 log 機率。它走的是詞彙空間，和 projector 的打分是兩條不同的路 → 論文第 461 行說它「not explicitly aligned with the scoring objective」。

> [!warning] COALA 官方 code 沒有實作 baseline
> `Guo0911/COALA` 的 `losses.py` 只有 MPD、DPD。定義來源是 [Huang & Wang 2025](https://arxiv.org/html/2510.23849v1)。

## 驗證

| 測試 | 預期 | 結果 |
|---|---|---|
| 圖 2b 例子 | 0.1 × 5 + 0.9 × 1.150889 = 1.535800 | ✅ |
| 只有 `<unbiased>` | 0.9 × 5.150889 = 4.635800 | ✅ |

呼叫：[[disc_loss()]]；被 [[losses.main()]] 呼叫；之後實驗 B。
