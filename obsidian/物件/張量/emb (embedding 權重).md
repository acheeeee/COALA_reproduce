---
tags: [COALA, 物件, 張量]
類別: 張量
型別: torch.nn.Parameter
shape: (49152, 576)
dtype: float32
---
# emb (embedding 權重)

← [[物件 總覽]]

**存什麼**：LM 的 embedding 查表矩陣 —— 第 k 列就是 token id k 的向量。

| 項目 | 內容 |
|---|---|
| shape | `(49152, 576)` = (vocab_size, [[D]]) |
| 定義於 | [[check_shape.main()]]：`lm.get_input_embeddings().weight` |

[[check_shape.py]] 讀它是為了確認兩件事：vocab = 49152、**hidden size D = 576**（→ adapter 的輸出必須是 576 維）。

它是 [[embed_layer]] 這個 `nn.Embedding` 模組的權重；凍結，不訓練。
