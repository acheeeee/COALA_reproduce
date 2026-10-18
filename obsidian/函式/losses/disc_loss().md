---
tags: [COALA, 函式, 模型層, loss, Stage2]
種類: 函式
檔案: losses.py
---
# disc_loss()

← [[losses.py]] · [[函式 總覽]]

**功能**：論文 Eq. 3 的 discriminative loss（baseline）—— 對**全部** K 個候選做一次 softmax，取正樣本的平均負 log 機率。

$$\mathcal{L}_{disc} = \frac{-1}{|E^+|}\sum_{i \in E^+}\log\frac{e^{s_i}}{\sum_{j \in E} e^{s_j}}$$

```python
def disc_loss(s, pos):
    log_p = F.log_softmax(s, dim=0)
    log_p_pos = log_p[pos]
    loss = -log_p_pos.mean()
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
| `log_p` | `(K,)` | 全部候選的 log 機率（和為 log 1 = 0） |
| `log_p_pos` | `(|E⁺|,)` | 只取正樣本 |

## ⭐ 為什麼它有 gradient conflict（測試 4 實測）
```
            <unb>   Watson  solved  survey  collect
梯度:      +0.006  +0.360  −0.384  +0.016  +0.002
```
梯度下降會把梯度為正的分數**往下推**。Watson 是正樣本卻被往下推（+0.36）——因為全域 softmax 讓兩個正樣本**搶同一份機率**，Watson 已經拿走 86%，disc 要它讓一點給 solved。這就是論文說的 **inter-positive competition**。

**推論：下界是 ln|E⁺|。** 兩個正樣本最好也只能各拿 50% → loss ≥ ln 2 ≈ 0.69，永遠到不了 0；24 個 rare word 的句子下界 ln 24 ≈ 3.18。

## 和原版（Huang & Wang 2025）的差別
原版對正樣本**加總**，COALA Eq. 3 是**平均**（÷|E⁺|）。這裡照 COALA。

## 被誰呼叫
[[bias_loss()]]、[[losses.main()]]；之後 `train_stage2.py` 的實驗 A（disc 單獨）。

## 驗證
手算 1.150889 ✅；單一正樣本時 = [[mpd_loss()]] = 5.150889 ✅；極端分數 100.0（有限值）✅
