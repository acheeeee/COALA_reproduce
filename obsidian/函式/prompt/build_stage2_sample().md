---
tags: [COALA, 函式, 模型層, prompt, Stage2]
種類: 函式
檔案: prompt.py
---
# build_stage2_sample()

← [[prompt.py]] · [[函式 總覽]]

**功能**：Stage 2 的序列組裝 —— 一句話 + K 個候選 → **一段共用前綴** + **K 條只放 entity 的尾巴**。前綴和尾巴分開回傳，讓 8b 可以只算一次前綴（KV cache）。

```python
def build_stage2_sample(audio_tokens, candidates, tokenizer, embed_layer):
```

## 輸入 → 輸出

| 參數 | 型別 / shape | 物件 |
|---|---|---|
| `audio_tokens` | `(L', 576)` | [[audio_tok]] |
| `candidates` | `list[str]`，長度 K | [[full_list (E)]]（index 0 = `<unbiased>`） |
| `tokenizer` | | [[tokenizer]] |
| `embed_layer` | | [[embed_layer]] |
| **輸出** | `dict` | [[sample (stage2 序列)]] |

## 序列長相
```
前綴（K 條共用）:  [ seg1 (15) | audio (L') | 題目 (13) ]          → prefix_embeds (P, 576)
尾巴 i:                                      [ " watson" ]    → suffix_embeds[i] (S, 576)
```
- 題目 = [[STAGE2_INSTRUCTION]]，結尾是 `Text:`
- 實測（假音訊 89 個）：P = 15 + 89 + 13 = **117**

## 內部物件

| 步驟 | 物件 |
|---|---|
| 1 | [[seg1]] → [[seg1_ids]] → [[seg1_emb]]；`inst_ids` → `inst_emb` |
| 2 | [[prefix_embeds]] = `cat([seg1_emb, audio_tokens, inst_emb])` |
| 3 | `cand_ids`：K 個 `encode(" " + e)`，長度各不相同 |
| 4 | `K`、`lens`、`S = max(lens)`；建空的 [[suffix_embeds]]、[[suffix_mask]]、[[entity_ids]] |
| 5 | 逐個候選：查表 → 放進第 i 列前 n 格；mask 填 1；ids 填入 |

## 為什麼 entity 放最後
1. **causal attention**：每個位置只看得到前面。放最後，entity token 的 hidden state 才同時看得到音訊、題目、自己
2. **KV cache**：前綴不受後面接什麼影響 → K 條共用，只算一次
3. 不能把 121 個詞塞同一條：後面的詞會看到前面的詞（分數受排列順序影響），而且 N=5000 會超過 2048 的 context window

## 為什麼要 `" " + e`
BPE 把「空格 + 字」當成一個單位。整句裡 entity 前面有空格（`Text: watson`），分開 encode 時也要帶空格，token 才會和 LM 平常看到的一樣。**斷詞邊界檢查**：分開 encode 的結果 == 整句一起 encode 的尾段 → 4 個候選全 ✅。

## 實測（[[prompt.main()]]）

| 候選 | token 數 | ids |
|---|---|---|
| `<unbiased>` | 4 | `[2067, 419, 30104, 46]`（當普通文字切） |
| watson | 2 | `[267, 19201]`（全小寫 → 人名被切碎） |
| solved | 1 | `[14168]` |
| craswellers | 4 | `[1650, 292, 5333, 366]` |

## 可改進（不急）
`zeros((K, S, 576), dtype=torch.float32)` 寫死了 576 和 float32；可以照 [[pad_batch()]] 從 `prefix_embeds` 抄 `shape[1]` 和 `dtype`。

## 呼叫了誰
[[build_seg1()]]、[[encode()]]、[[embed_layer]]

## 被誰呼叫
[[prompt.main()]]；之後 `COALAModel.forward_stage2`（8b）。
