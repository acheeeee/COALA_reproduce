---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: list[dict]
---
# samples

← [[物件 總覽]]

**存什麼**：一個 batch 裡每句話的 [[sample (stage1 序列)]]，長度各不相同。

| 項目 | 內容 |
|---|---|
| 型別 | `list[dict]`，長度 B |
| 定義於 | [[COALAModel.forward_stage1()]] 的區域變數 |

```python
samples = []
for i in range(len(batch["text"])):
    audio_tok = ctc_compress(h[i], ctc_logits[i])
    samples.append(build_stage1_sample(audio_tok, entity_lists[i], batch["text"][i], ...))
```

→ [[pad_batch()]] → [[padded]]
