---
tags: [COALA, 物件, 資料結構, prompt]
類別: 資料結構
型別: str
---
# seg2

← [[物件 總覽]]

**存什麼**：prompt 的第 3 段文字 —— 指令 + entity 清單，結束 user turn 並開啟 assistant turn。

| 項目 | 內容 |
|---|---|
| 型別 | `str` |
| 由誰產生 | [[build_seg2()]] |

```
Transcribe the speech. The following entities may appear: apostle, gospel, quilter, ....<|im_end|>
<|im_start|>assistant
```
（[[entity_list]] 為空時第一行變成 `Transcribe the speech.`）

**最後一行 `<|im_start|>assistant\n` 很重要**：生成時 LM 看到它，就知道接下來輪到自己說話。

由 [[instruction]] 組成 → [[seg2_ids]] → [[seg2_emb]]
