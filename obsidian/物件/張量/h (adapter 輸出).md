---
tags: [COALA, 物件, 張量, 音訊]
類別: 張量
型別: torch.Tensor
shape: (B, 1500/stack, 576)
dtype: float32
aliases: [h, hidden]
---
# h (adapter 輸出)

← [[物件 總覽]]

**存什麼**：投影到 LM embedding 空間的音訊向量。同一份資料走兩條路：去算 CTC，也被挑選後餵給 LM。

| 項目 | 內容 |
|---|---|
| 型別 | `torch.Tensor` |
| shape | `(B, 1500/stack, 576)` → **stack=2：`(B, 750, 576)`**；stack=4：`(B, 375, 576)` |
| dtype | `float32`，可訓練（帶梯度） |
| 別名 | [[ctc_compress()]] 的參數 `hidden`（單句 `(L, 576)` = `h[i]`）、[[CTCHead.forward()]] 的參數 `h` |

> 和 [[h (projector 輸入)]] 是不同的東西（那個是 LM 輸出的 hidden state）。

## 維度意義

| 維度 | 意義 |
|---|---|
| 1500/stack | 「位置」數。stack=2 → 25 位置/秒；stack=4 → 12.5 位置/秒 |
| 576 | = [[D]]，LM 的 hidden size，這樣才能和文字 embedding 拼在一起 |

## 從哪來 → 到哪去
```
x (encoder 輸出) ──AudioAdapter──→ h ──┬─CTCHead──→ ctc_logits
                                              └─ctc_compress()（用 ctc_logits 挑位置）──→ audio_tok
```
圖中物件：[[x (encoder 輸出)]]、[[AudioAdapter]]、[[CTCHead]]、[[ctc_logits]]、[[ctc_compress()]]、[[audio_tok]]
- 產生：[[AudioAdapter.forward()]]，在 [[COALAModel.encode_audio()]] 中呼叫
- 使用：[[CTCHead.forward()]]、[[ctc_compress()]]

> [!important] 餵給 LM 的是 576 維的 h，不是 29 維的 logits
> logits 只負責「決定留哪些位置」。576 維保有語者、語調等聲學細節，壓成 29 維就退化成先轉文字再處理。
