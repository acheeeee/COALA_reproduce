---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: str
---
# seg1

← [[物件 總覽]]

**存什麼**：prompt 的第 1 段文字 —— system 訊息 + user turn 的開頭。音訊會緊接在它後面。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 由誰產生 | [[build_seg1()]] |
| token 數 | 16 |

```
<|im_start|>system
You are a speech recognition assistant.<|im_end|>
<|im_start|>user
```
（`<|im_start|>` = `tokenizer.bos_token`、`<|im_end|>` = `tokenizer.eos_token`，SmolLM2 的 chat template 格式）

**對每一句話都完全相同** → [[seg1_ids]] → [[seg1_emb]]

由 [[SYSTEM_PROMPT]] 組成。
