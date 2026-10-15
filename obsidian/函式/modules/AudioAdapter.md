---
tags: [COALA, 函式, 模型層, 類別]
種類: 類別
檔案: modules.py
---
# AudioAdapter

← [[modules.py]] · [[函式 總覽]]

**功能**：把 Whisper encoder 的輸出（1280 維、1500 步）接到 LM 的 embedding 空間（576 維），同時把時間長度縮短 `stack` 倍。

```python
class AudioAdapter(nn.Module):
```

## 方法

| 方法 | 做什麼 |
|---|---|
| [[AudioAdapter.__init__()]] | 建一層 `Linear(1280·stack, 576)` |
| [[AudioAdapter.forward()]] | frame stacking + 投影 |

## 屬性

| 屬性 | 型別 | 內容 |
|---|---|---|
| `self.stack` | `int` | 降採樣倍率，**預設 2** |
| `self.proj` | `nn.Linear(1280·stack, 576)` | 唯一的可訓練層 |

## 參數量

| stack | Linear 輸入 | 參數量 | 佔 5.7M 預算 |
|---|---|---|---|
| 2 | 2560 | **1,475,136** | 26% |
| 4 | 5120 | 2,949,696 | 52% |

## 實例
[[COALAModel]] 的 `self.adapter`

> [!important] 論文完全沒給 adapter 架構（疑點 #5）
> stack 和 CTC 詞彙單位有耦合約束：**位置數/秒 ≥ CTC 單位數/秒**。字元級 CTC（最快 23 字/秒）配 stack=4（12.5 位置/秒）→ 83.2% 句子 CTC 無解。stack=2（25 位置/秒）→ 0%。
> ⚠️ 原始碼的 docstring 還寫著 `(B,1500,1280) → (B,375,576)`、「相鄰 4 個 frame」，是 stack=4 時代的舊說明。
