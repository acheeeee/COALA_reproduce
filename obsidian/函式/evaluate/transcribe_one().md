---
tags: [COALA, 函式, 評估]
種類: 函式
檔案: evaluate.py
---
# transcribe_one()

← [[evaluate.py]] · [[函式 總覽]]

**功能**：對一句話做 greedy 生成，回傳轉寫文字。

```python
@torch.no_grad()
def transcribe_one(model, input_features, entity_list, dev):
```

## 輸入 → 輸出

| 參數 | 物件 |
|---|---|
| `model` | [[COALAModel]] 實例 |
| `input_features` | [[input_features]] `(80, 3000)`，單句 |
| `entity_list` | [[entity_list]] |
| `dev` | [[dev]] |
| **輸出** | [[hyp]]（`str`） |

## 資料流
```
input_features (80,3000) ──unsqueeze(0)──→ feats (1,80,3000)
  ──COALAModel.encode_audio()──→ h, ctc_logits
  ──ctc_compress()(h[0], ctc_logits[0])──→ audio_tok
  ──build_gen_prompt()──→ prompt (生成 prompt)
  ──model.lm.generate(...)──→ out_ids
  ──tokenizer.decode(skip_special_tokens=True)──→ text ──.strip()──→ hyp
```
圖中物件：[[COALAModel.encode_audio()]]、[[ctc_compress()]]、[[audio_tok]]、[[build_gen_prompt()]]、[[prompt (生成 prompt)]]、[[out_ids]]

## generate 的設定

| 參數 | 值 | 意義 |
|---|---|---|
| `max_new_tokens` | [[MAX_NEW_TOKENS]] = 200 | 上限 |
| `do_sample=False`、`num_beams=1` | greedy | 每步取機率最高的 token，結果可重現 |
| `eos_token_id` | `<|im_end|>` | 看到它就停 |
| `pad_token_id` | 同 eos | 避免警告 |

> [!bug] Bug A：要把 `.strip()` 改成 `" ".join(text.split())`
> 生成的文字可能含 `\n` → TSV 一句變兩行。

被 [[evaluate.main()]] 呼叫。
