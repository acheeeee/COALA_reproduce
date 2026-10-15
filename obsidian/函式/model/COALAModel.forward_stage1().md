---
tags: [COALA, 函式, 模型層, loss]
種類: 方法
檔案: model.py
---
# COALAModel.forward_stage1()

← [[COALAModel]] · [[model.py]] · [[函式 總覽]]

**功能**：Stage 1（ASR）的完整 forward —— 一個 batch 進來，CTC loss + LM 轉寫 loss 出去。

```python
def forward_stage1(self, batch, entity_lists):
```

## 輸入 → 輸出

| 參數 | 物件 |
|---|---|
| `batch` | [[batch]]（需要 `input_features`、`ctc_ids`、`ctc_lens`、`text`） |
| `entity_lists` | [[entity_lists]]（每句 10 個） |
| **輸出** | [[out (stage1 loss dict)]] |

## 步驟與內部物件

| # | 做什麼 | 產生的物件 |
|---|---|---|
| 1 | `feats = batch["input_features"]` → [[COALAModel.encode_audio()]] | [[h (adapter 輸出)]]、[[ctc_logits]] |
| 2 | [[compute_ctc_loss()]] | [[ctc_loss]] |
| 3 | `embed_layer = self.lm.get_input_embeddings()` | [[embed_layer]] |
| 4 | **逐句**：[[ctc_compress()]] → [[build_stage1_sample()]] | [[audio_tok]] → [[sample (stage1 序列)]]，收進 [[samples]] |
| 5 | [[pad_batch()]] | [[padded]] |
| 6 | `out = self.lm(**padded)` | [[out (LM 輸出物件)]] → `out.loss` = [[ce_loss]] |
| 7 | `total = 0.3·ctc + 1.0·ce` | [[loss (total)]] |

```python
samples = []
embed_layer = self.lm.get_input_embeddings()
for i in range(len(batch["text"])):
    audio_tok = ctc_compress(h[i], ctc_logits[i])
    samples.append(build_stage1_sample(audio_tok, entity_lists[i], batch["text"][i],
                                       self.tokenizer, embed_layer))
padded = pad_batch(samples)
out = self.lm(**padded)
ce_loss = out.loss
total = CTC_WEIGHT * ctc_loss + CE_WEIGHT * ce_loss
return dict(loss=total, ctc_loss=ctc_loss, ce_loss=ce_loss)
```

用到的常數：[[CTC_WEIGHT]]、[[CE_WEIGHT]]

## 被誰呼叫
[[train_stage1.main()]]、[[validate()]]、[[overfit_test.main()]]、[[model.main()]]

> [!important] 難點：中間那段「逐句處理再 pad 回 batch」
> [[ctc_compress()]] 的輸出長度隨內容而變 → batch 在這裡被拆開，處理完再用 [[pad_batch()]] 組回去。
