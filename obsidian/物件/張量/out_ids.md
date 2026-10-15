---
tags: [COALA, 物件, 張量]
類別: 張量
型別: torch.Tensor
shape: (1, n_new)
dtype: int64
---
# out_ids

← [[物件 總覽]]

**存什麼**：LM 生成出來的轉寫 token id。

| 項目 | 內容 |
|---|---|
| shape | `(1, n_new)`，n_new ≤ [[MAX_NEW_TOKENS]] = 200 |
| 定義於 | [[transcribe_one()]] |

```python
out_ids = model.lm.generate(inputs_embeds=..., attention_mask=...,
                            max_new_tokens=200, do_sample=False, num_beams=1, ...)
text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
```

只給 `inputs_embeds`（沒給 `input_ids`）時，HF 回傳的**只有新產生的 token**，不含 prompt。

→ decode 成 [[hyp]]

> [!bug] 已知問題（[[evaluate.py]] Bug A）
> decode 出來的文字可能含 `\n`，寫進 TSV 會多出行。要改成 `" ".join(text.split())`。
