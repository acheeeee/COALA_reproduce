---
tags: [COALA, 函式, 模型層, loss, Stage2]
種類: 函式
檔案: losses.py
---
# mpd_loss()

← [[losses.py]] · [[函式 總覽]]

**功能**：論文 Eq. 4 的 MPD-Loss —— 每個正樣本**各自**只和 E⁻ 做 softmax，正樣本之間不再互搶。

$$\mathcal{L}_{MPD} = \frac{-1}{|E^+|}\sum_{i \in E^+}\log\frac{e^{s_i}}{e^{s_i} + \sum_{j \in E^-} e^{s_j}}$$

```python
def mpd_loss(s, pos):
    pos_scores = s[pos]
    neg_scores = s[~pos]
    lse_neg = torch.logsumexp(neg_scores, dim=0)       # log Σ exp(s_j)，純量
    lse_pair = torch.logaddexp(pos_scores, lse_neg)    # log(exp(s_i) + exp(lse_neg))，(|E⁺|,)
    loss = (lse_pair - pos_scores).mean()
    return loss
```

## 輸入 → 輸出

| 參數 | shape | 物件 |
|---|---|---|
| `s` | `(K,)` | [[s (entity 分數)]] |
| `pos` | `(K,)` bool | [[pos (E⁺ 遮罩)]] |
| **輸出** | `()` | 純量 loss |

## 內部物件

| 名稱 | shape | 內容 |
|---|---|---|
| `pos_scores` / `neg_scores` | `(|E⁺|,)` / `(|E⁻|,)` | `s[pos]` / `s[~pos]` |
| `lse_neg` | `()` | 所有負樣本的 log-sum-exp |
| `lse_pair` | `(|E⁺|,)` | 每個正樣本 + 負樣本總和 的 log-sum-exp（broadcasting 一次算完） |

## 為什麼這樣寫等於公式
`log(e^{sᵢ} / (e^{sᵢ} + Σe^{sⱼ})) = sᵢ − log(e^{sᵢ} + e^{lse_neg})` → 取負號就是 `lse_pair − pos_scores`。全程沒有直接算 `exp`，所以 s = 300 也不會溢位。

## 梯度（測試 4）
```
            <unb>   Watson  solved  survey  collect
梯度:      +0.024  −0.013  −0.085  +0.065  +0.009
```
兩個正樣本都是負梯度 → **一起被往上推**，沒有衝突。

## 和官方 code 對照
官方 `recipes/models/modules/losses.py` 的 `MPD_loss` 用 for 迴圈逐個正樣本算 `logsumexp(stack([pos, neg_lse]))`，數學上完全相同。

被 [[losses.main()]] 呼叫；之後實驗 C。
