---
tags: [COALA, 函式, 模型層, 類別, Stage2]
種類: 類別
檔案: projector.py
---
# DiscriminativeProjector

← [[projector.py]] · [[函式 總覽]]

**功能**：論文 Eq. 1、2 —— 把 LM 對每個 entity token 的 hidden state 轉成一個匹配分數，再對 entity 的所有 token 做長度正規化平均。

```python
class DiscriminativeProjector(nn.Module):
```

## 方法

| 方法 | 做什麼 |
|---|---|
| [[DiscriminativeProjector.__init__()]] | 建兩層 MLP |
| [[DiscriminativeProjector.forward()]] | `(N, L, 576)` → `(N,)` |

## 屬性
`self.mlp` = `Sequential(Linear(576, 576), ReLU(), Linear(576, 1))`

## 參數量
`D′(D + 2) + 1` = 576 × 578 + 1 = **332,929**（[[projector.main()]] 實測吻合）

## 在整個模型中的位置（Stage 2，尚未接上）
```
LM（LoRA scoring）最後一層 hidden state → h (projector 輸入) (121, L, 576)
  → DiscriminativeProjector → s (entity 分數) (121,)
  → losses.py（disc / MPD / DPD）
```
圖中物件：[[h (projector 輸入)]]、[[s (entity 分數)]]
Stage 2 只訓練它 + LoRA scoring。

> 論文：這個 MLP 讓匹配強度「independent of the standard language model vocabulary distribution」—— 不走 49152 維的詞彙 softmax，而是另外學一個專門的判別空間。
