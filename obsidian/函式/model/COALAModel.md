---
tags: [COALA, 函式, 模型層, 類別]
種類: 類別
檔案: model.py
---
# COALAModel

← [[model.py]] · [[函式 總覽]]

**功能**：組裝者 —— 把 Whisper encoder、AudioAdapter、CTCHead、SmolLM2 + 兩組 LoRA、tokenizer 組成完整模型。目前只有 Stage 1 的 forward。

```python
class COALAModel(nn.Module):
```

## 方法

| 方法 | 做什麼 |
|---|---|
| [[COALAModel.__init__()]] | 載入並組裝五個零件、設定凍結 |
| [[COALAModel.encode_audio()]] | 音訊 → [[h (adapter 輸出)]] + [[ctc_logits]] |
| [[COALAModel.forward_stage1()]] | 一個 batch → 三個 loss |
| （尚未實作）`forward_stage2` | 音訊 + 121 個候選 → 121 個分數 |

## 屬性（零件）

| 屬性 | 物件 / 類別 | 可訓練 | 參數量 |
|---|---|---|---|
| `self.encoder` | [[encoder]] | ❄️ | 636.8M |
| `self.adapter` | [[AudioAdapter]] | 🔥 | 1,475,136（stack=2） |
| `self.ctc_head` | [[CTCHead]] | 🔥 | 16,733 |
| `self.lm` | [[lm]]（LoRA asr 🔥 / scoring） | 部分 | 134.5M + 921,600 × 2 |
| `self.tokenizer` | [[tokenizer]] | — | — |

Stage 1 可訓練合計 **2,413,469**（stack=2）。

## 實例（`model`）
[[train_stage1.main()]]、[[evaluate.main()]]、[[overfit_test.main()]]、[[model.main()]]

## 資料流
```
input_features → encode_audio → h, ctc_logits ─┬→ compute_ctc_loss → ctc_loss
                                               └→ ctc_compress（逐句）→ build_stage1_sample → pad_batch
                                                  → lm(**padded) → ce_loss
total = 0.3·ctc + 1.0·ce
```
