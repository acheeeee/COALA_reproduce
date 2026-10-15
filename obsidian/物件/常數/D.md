---
tags: [COALA, 物件, 常數]
類別: 常數
型別: int
---
# D

← [[物件 總覽]]

**存什麼**：LM 的 hidden size，也就是論文 Eq. 2 的 `D`。整個模型「進入 LM 之後」的向量維度都是它。

| 項目 | 內容 |
|---|---|
| 型別 | `int` |
| 值 | `576` |
| 定義於 | [[projector.py]]（其他檔案用字面值 `576`） |
| 來源 | SmolLM2 `config.hidden_size`，由 [[check_models.py]]、[[check_shape.py]] 實測 |

## 576 出現在哪些地方

| 物件 | 為什麼是 576 |
|---|---|
| [[h (adapter 輸出)]] | adapter 要把音訊投影到 LM 的 embedding 空間 |
| [[seg1_emb]] / [[seg2_emb]] / [[target_emb]] | embedding 查表的輸出 |
| [[inputs_embeds]] | LM 的輸入 |
| [[h (projector 輸入)]] | LM 最後一層 hidden state |
| [[emb (embedding 權重)]] | `(49152, 576)` |

## 用在哪裡（以常數形式）
- [[DiscriminativeProjector.__init__()]] 的 `d` 預設值
