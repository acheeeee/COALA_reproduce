---
tags: [COALA, 函式, 模型層, prompt]
種類: 函式
檔案: prompt.py
---
# build_stage1_sample()

← [[prompt.py]] · [[函式 總覽]]

**功能**：把一句話的音訊 token、entity、逐字稿組成 LM 訓練用的三件組（inputs_embeds / labels / attention_mask）。

```python
def build_stage1_sample(audio_tokens, entity_list, target_text, tokenizer, embed_layer):
```

## 輸入 → 輸出

| 參數 | 型別 / shape | 物件 |
|---|---|---|
| `audio_tokens` | `(L', 576)` | [[audio_tok]] |
| `entity_list` | `list[str]` | [[entity_list]] |
| `target_text` | `str` | [[text]] |
| `tokenizer` | | [[tokenizer]] |
| `embed_layer` | `nn.Embedding` | [[embed_layer]] |
| **輸出** | `dict` | [[sample (stage1 序列)]] |

## 內部物件（依序產生）

| 步驟 | 物件 |
|---|---|
| 0 | [[dev]] = `audio_tokens.device` |
| 1 文字 | [[seg1]]、[[seg2]]、[[target]] |
| 2 → id | [[seg1_ids]]、[[seg2_ids]]、[[target_ids]] |
| 3 → 向量 | [[seg1_emb]]、[[seg2_emb]]、[[target_emb]] |
| 4 拼接 | [[inputs_embeds]] = `cat([seg1_emb, audio_tokens, seg2_emb, target_emb])` |
| 5 長度 | `n1, L, n2, m`（一律用 `.shape[0]` / `len(tensor)`） |
| 6 答案 | [[labels]] = `cat([full(n1+L+n2, -100), target_ids])` |
| 7 遮罩 | [[attention_mask]] = `ones(n1+L+n2+m)` |

## 長相
```
inputs_embeds: [ seg1_emb | audio_tok | seg2_emb | target_emb ]
labels:        [   -100   |   -100    |   -100   | target_ids ]
attention:     [    1     |     1     |    1     |     1      ]
```

## 呼叫了誰
[[build_seg1()]]、[[build_seg2()]]、[[encode()]]、[[embed_layer]]

## 被誰呼叫
[[COALAModel.forward_stage1()]]（每句一次）、[[prompt.main()]]

> [!danger] 長度用 token 數，不是字元數
> 曾經用 `len(target_text)`（字元數）→ attention_mask 長 209、inputs_embeds 長 160 對不上。
