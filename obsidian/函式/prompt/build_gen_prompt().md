---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# build_gen_prompt()

← [[prompt.py]] · [[函式 總覽]]

**功能**：組出「生成用」的 prompt —— 和 [[build_stage1_sample()]] 一樣，但**沒有 target**、沒有 labels，並加上 batch 維。

```python
def build_gen_prompt(audio_tokens, entity_list, tokenizer, embed_layer):
```

## 輸入 → 輸出

| 參數 | 物件 |
|---|---|
| `audio_tokens` | [[audio_tok]] `(L', 576)` |
| `entity_list` | [[entity_list]] |
| `tokenizer` | [[tokenizer]] |
| `embed_layer` | [[embed_layer]] |
| **輸出** | [[prompt (生成 prompt)]] |

## 內部物件
[[dev]]、[[seg1]]、[[seg2]]、[[seg1_ids]]、[[seg2_ids]]、[[seg1_emb]]、[[seg2_emb]]、[[inputs_embeds]]、`total`、[[attention_mask]]

```python
inputs_embeds = torch.cat([seg1_emb, audio_tokens, seg2_emb], dim=0).unsqueeze(0)
attention_mask = torch.ones(1, inputs_embeds.shape[1], dtype=torch.long, device=dev)
```
`.unsqueeze(0)`：在最前面加一個大小 1 的維度 → `(1, total, 576)`

## 呼叫了誰
[[build_seg1()]]、[[build_seg2()]]、[[encode()]]、[[embed_layer]]

## 被誰呼叫
[[transcribe_one()]]
