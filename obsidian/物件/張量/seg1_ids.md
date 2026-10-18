---
tags: [COALA, 物件, 張量, prompt]
類別: 張量
型別: torch.Tensor
shape: (n1,)
dtype: int64
---
# seg1_ids

← [[物件 總覽]]

**存什麼**：[[seg1]] 文字（system 訊息 + user turn 開頭）的 token id。

| 項目 | 內容 |
|---|---|
| shape | `(n1,)`，實測 **n1 = 15** |
| dtype | `torch.long` |
| 定義於 | [[build_stage1_sample()]]、[[build_gen_prompt()]] |

```python
seg1_ids = encode(seg1, tokenizer, dev)      # add_special_tokens=False
```

→ [[embed_layer]] 查表 → [[seg1_emb]]

> [!tip] Stage 2 的 KV cache 關鍵
> seg1 對**每一句、每一個候選**都完全相同，Stage 2 再加上 [[audio_tok]] 和題目（15 + 89 + 13 = 117 token，見 [[prefix_embeds]]）在同一句的 121 次 forward 中也相同 → 可以只算一次。
