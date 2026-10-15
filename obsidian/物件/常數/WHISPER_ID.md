---
tags: [COALA, 物件, 常數]
類別: 常數
型別: str
---
# WHISPER_ID

← [[物件 總覽]]

**存什麼**：Whisper 預訓練模型在 HuggingFace Hub 上的名稱。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 值 | `"openai/whisper-large-v2"` |
| 定義於 | [[dataset.py]]、[[model.py]]、[[check_shape.py]]（三處各自重複定義，值相同） |

## 用在哪裡

| 誰用 | 拿來做什麼 |
|---|---|
| [[COALAModel.__init__()]] | `WhisperModel.from_pretrained(WHISPER_ID)` → 取出 [[encoder]] |
| [[train_stage1.main()]]、[[evaluate.main()]]、[[overfit_test.main()]]、[[model.main()]]、[[dataset.main()]] | `WhisperFeatureExtractor.from_pretrained(WHISPER_ID)` → 建出 [[fe]] |
| [[check_shape.main()]] | 同時載入 [[fe]] 與 encoder 量 shape |

> [!note] 訓練腳本是從 `model.py` import 這個常數（`from model import COALAModel, WHISPER_ID`），所以真正「生效」的是 [[model.py]] 那份。
