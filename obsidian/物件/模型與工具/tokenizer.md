---
tags: [COALA, 物件, 模型與工具, prompt]
類別: 模型與工具
型別: transformers.AutoTokenizer（SmolLM2）
aliases: [tok, self.tokenizer]
---
# tokenizer

← [[物件 總覽]]

**是什麼**：SmolLM2 的 BPE tokenizer，負責「文字 ↔ token id」。

| 項目 | 內容 |
|---|---|
| 建立 | `AutoTokenizer.from_pretrained(LM_ID)`，在 [[COALAModel.__init__()]] 存成 `self.tokenizer` |
| 詞彙量 | 49,152 |
| `bos_token` | `<|im_start|>`（chat template 的「開始一段對話」） |
| `eos_token` | `<|im_end|>`（「這段對話結束」，也是生成的停止符號） |
| 別名 | `tok`（[[prompt.main()]]） |

## 三種用法

| 用法 | 在哪 |
|---|---|
| `tokenizer(text, add_special_tokens=False)["input_ids"]` → id | [[encode()]] |
| `tokenizer.bos_token` / `eos_token` 字串拼模板 | [[build_seg1()]]、[[build_seg2()]]、[[build_stage1_sample()]] |
| `tokenizer.decode(ids, skip_special_tokens=True)` → 文字 | [[transcribe_one()]] |

> [!note] 為什麼 `add_special_tokens=False`
> 模板裡已經手動放了 `<|im_start|>` / `<|im_end|>`，再讓 tokenizer 自動加會重複。

## 注意
它只管**文字**。音訊向量 [[audio_tok]] 沒有 token id，不經過 tokenizer。
