---
tags: [COALA, 函式, 模型層, loss, Stage2]
種類: 函式
檔案: losses.py
---
# dpd_loss()

← [[losses.py]] · [[函式 總覽]]

**功能**：論文 Eq. 5–7 的 DPD-Loss —— 每個 entity 各自做一個二元分類（在 / 不在），正負樣本完全解耦。

$$\mathcal{L}_{DPD} = \underbrace{\frac{-1}{|E^+|}\sum_{i\in E^+}\log\sigma(s_i)}_{\mathcal{P}(E^+)} + \underbrace{\frac{-1}{|E^-|}\sum_{j\in E^-}\log(1-\sigma(s_j))}_{\mathcal{N}(E^-)}$$

```python
def dpd_loss(s, pos):
    pos_scores = s[pos]
    neg_scores = s[~pos]
    loss_pos = -F.logsigmoid(pos_scores).mean()     # 𝒫(E⁺)
    loss_neg = -F.logsigmoid(-neg_scores).mean()    # 𝒩(E⁻)，用 log(1−σ(x)) = logσ(−x)
    return loss_pos + loss_neg
```

## 輸入 → 輸出

| 參數 | shape | 物件 |
|---|---|---|
| `s` | `(K,)` | [[s (entity 分數)]] |
| `pos` | `(K,)` bool | [[pos (E⁺ 遮罩)]] |
| **輸出** | `()` | 純量 loss |

## 內部物件
`pos_scores`、`neg_scores`、`loss_pos`（𝒫）、`loss_neg`（𝒩）

## 重點
- `log(1 − σ(x)) = logσ(−x)`：避免 σ(x) ≈ 1 時算 `log(0)`
- **兩項分開平均**：E⁻ 有 ~118 個、E⁺ 只有幾個，分開平均讓兩邊權重相當
- 優化的是**絕對**分數（σ(s) 接近 1 或 0），不是相對排名 → 推論時可以直接用 s > 0 當門檻

## 梯度（測試 4）
```
            <unb>   Watson  solved  survey  collect
梯度:      +0.040  −0.024  −0.135  +0.090  +0.016
```

## 和官方 code 對照
官方 `DPD_loss` 用 `F.binary_cross_entropy_with_logits(pos, ones)` + `(neg, zeros)`，各自取平均 → 和這裡完全相同。官方 `stage_2.sh` 的預設就是 `DPD-loss`。

被 [[losses.main()]] 呼叫；之後實驗 D。
