---
tags: [COALA, 物件, 模型與工具, prompt]
類別: 模型與工具
型別: torch.nn.Embedding(49152, 576)
aliases: [embed]
---
# embed_layer

← [[物件 總覽]]

**是什麼**：LM 的 embedding 查表層 —— 給 token id，回傳 576 維向量。凍結。

| 項目 | 內容 |
|---|---|
| 型別 | `nn.Embedding(49152, 576)` |
| 取得 | `self.lm.get_input_embeddings()` |
| 權重 | [[emb (embedding 權重)]] `(49152, 576)` |
| 別名 | `embed`（[[prompt.main()]]） |

## 用法
```python
seg1_emb = embed_layer(seg1_ids)      # (n1,) long → (n1, 576)
```
用在 [[build_stage1_sample()]]、[[build_gen_prompt()]]：把 [[seg1_ids]]、[[seg2_ids]]、[[target_ids]] 變成 [[seg1_emb]]、[[seg2_emb]]、[[target_emb]]。

## 為什麼要自己查表
一般用 `lm(input_ids=...)` 時 LM 會自己查。但這裡要把**沒有 token id 的** [[audio_tok]] 插在中間，只能自己先把文字查成向量，再 `torch.cat`，最後走 `lm(inputs_embeds=...)`。
