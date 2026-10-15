---
tags: [COALA, 函式, 工具]
種類: main
檔案: check_shape.py
---
# check_shape.main()

← [[check_shape.py]] · [[函式 總覽]]

**功能**：逐步印出音訊從波形到 encoder 輸出的 shape，並確認 LM embedding 維度。

## 步驟
1. 建 [[fe]]、[[encoder]]（`.eval()`）
2. 載入 LM，取 [[emb (embedding 權重)]] → 印 `(49152, 576)` → adapter 輸出必須是 576
3. 對 [[pack_two_utterances()]] 的兩句：

| 步 | 物件 | shape |
|---|---|---|
| [1] `sf.read` | [[wav]]（變數名 `wave`）、`sr` | `(秒數 × 16000,)`、16000 |
| [2] [[fe]] | [[input_features]]（變數名 `mel`） | `(1, 80, 3000)` |
| [3] encoder（`no_grad`） | [[x (encoder 輸出)]]（變數名 `hid`） | `(1, 1500, 1280)` |

## 結論
最短句和最長句的 [2]、[3] **完全一樣** → 確認固定 30 秒 pad；1280 = adapter 的輸入維度。

## 用到的常數
[[MANIFEST]]、[[WHISPER_ID]]、[[LM_ID]]
