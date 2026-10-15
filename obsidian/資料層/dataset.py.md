---
tags: [COALA, 資料層, 檔案]
層: 資料層
狀態: 完成
---

# dataset.py

← [[資料層 總覽]] · [[COALA 索引]]

執行期使用。讀 [[build_manifest.py]] 的產物，被 [[train_stage1.py]]、[[evaluate.py]]、[[overfit_test.py]] 使用。


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：PyTorch 資料管線 —— manifest 一列 → 一筆模型輸入（log-Mel + CTC id），再打包成 batch。

| 類型 | 內容 |
|---|---|
| import 外部 | `json`、`pathlib`、`soundfile`、`torch`、`torch.utils.data`、`transformers`（`WhisperFeatureExtractor`；`AutoTokenizer` import 了但沒用） |
| import 自己的 | 無 |
| 被誰 import | [[train_stage1.py]]、[[evaluate.py]]、[[overfit_test.py]]、[[model.py]]（main） |

**定義的函式 / 類別**

| 名稱 | 一句話 |
|---|---|
| [[text_to_ctc_ids()]] | 文字 → CTC id list |
| [[LibriSpeechDataset]] | Dataset 類別 |
| ↳ [[LibriSpeechDataset.__init__()]] | 讀 manifest、過濾 >30 秒 |
| ↳ [[LibriSpeechDataset.__len__()]] | 句數 |
| ↳ [[LibriSpeechDataset.__getitem__()]] | 讀音檔 → [[sample (Dataset 單筆)]] |
| [[collate_fn()]] | B 筆 → [[batch]] |
| [[dataset.main()]] | shape 檢查 |

**定義的常數**：[[WHISPER_ID]]、[[LM_ID]]（未使用）、[[MAX_DURATION]]、[[CTC_CHARS]]、[[CTC_BLANK_ID]]、[[CTC_CHAR2ID]]

**資料流**：
```
manifest TSV → __init__ → self.rows
ds[i]：audio_path → wav → fe → input_features；text → ctc_ids
collate_fn：→ batch（input_features 疊起來、ctc_ids 補 0、多出 ctc_lens）
```
圖中物件：[[manifest TSV]]、[[wav]]、[[fe]]、[[input_features]]、[[text]]、[[ctc_ids]]、[[batch]]、[[ctc_lens]]
<!-- 結構:end -->

---

## 功能

把 manifest 的一行變成模型能吃的 tensor。

---

## 函式與類別

| 函式 / 類別 | 輸入 | 輸出 |
|---|---|---|
| `text_to_ctc_ids(text)` | `text: str`（小寫） | `list[int]` — 跳過表外字元 |
| `LibriSpeechDataset.__init__(manifest_path, feature_extractor, max_duration=30.0)` | manifest 路徑、`WhisperFeatureExtractor` | — （過濾 `duration > 30` 並印跳過數） |
| `LibriSpeechDataset.__len__()` | 無 | `int` |
| `LibriSpeechDataset.__getitem__(i)` | `i: int` | `dict` ↓ |
| `collate_fn(batch)` | `list[dict]` | `dict` ↓ |

### `__getitem__` 回傳

| 鍵 | 型別 / shape |
|---|---|
| `utt_id` | `str` |
| `input_features` | `(80, 3000)` tensor ← log-Mel |
| `text` | `str` |
| `ctc_ids` | `(n_chars,)` long tensor |
| `rare_words` | `list[str]` |

```python
def __getitem__(self, i):
    r = self.rows[i]
    wav, sr = sf.read(r["audio_path"])
    out = self.fe(wav, sampling_rate=sr, return_tensors="pt")
    input_features = out.input_features[0]       # [0] 去掉 batch 維 → (80, 3000)
    ctc_ids = torch.tensor(text_to_ctc_ids(r["text"]), dtype=torch.long)
    return {"utt_id": r["utt_id"], "input_features": input_features,
            "text": r["text"], "ctc_ids": ctc_ids,
            "rare_words": r["rare_words"]}
```

`self.fe(...)` 會回傳一個**帶 batch 維**的結果 `(1, 80, 3000)`，因為 feature extractor 設計上支援一次處理多筆。這裡一次只處理一句，所以用 `[0]` 取出來 → `(80, 3000)`。batch 維之後由 `collate_fn` 統一加回去。

### `collate_fn` 回傳

| 鍵 | 型別 / shape |
|---|---|
| `utt_id` | `list[str]` |
| `input_features` | `(B, 80, 3000)` |
| `text` | `list[str]` |
| `ctc_ids` | `(B, max_len)` ← padding 補 0 |
| `ctc_lens` | `(B,)` ← **真實長度** |
| `rare_words` | `list[list[str]]` |

```python
def collate_fn(batch):
    batch_list = [b["input_features"] for b in batch]
    input_features = torch.stack(batch_list)              # (B, 80, 3000)

    ctc_lens = torch.tensor([len(b["ctc_ids"]) for b in batch], dtype=torch.long)
    max_len = int(ctc_lens.max())
    ctc_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        ctc_ids[i, : len(b["ctc_ids"])] = b["ctc_ids"]
    ...
```

`torch.stack` vs `torch.cat` 的差別：**`stack` 會新增一個維度**（N 個 `(80,3000)` → `(N,80,3000)`），`cat` 是在現有維度上接起來（N 個 `(80,3000)` 沿 dim=0 接 → `(80N, 3000)`）。這裡要的是前者。

---

## CTC 詞彙表

```python
CTC_CHARS    = " 'abcdefghijklmnopqrstuvwxyz"     # 28 個
CTC_BLANK_ID = 0
CTC_CHAR2ID  = {c: i + 1 for i, c in enumerate(CTC_CHARS)}   # id 0 留給 blank
```

**用實際資料確認過**：

```bash
awk -F'\t' 'NR>1 {print $5}' data/manifest/train-clean-100.tsv | fold -w1 | sort -u | tr -d '\n'
# 輸出： 'abcdefghijklmnopqrstuvwxyz
```

空格 + 撇號 + a-z = 28，**無數字、無其他標點**。加 blank 剛好 **29**，與 [[modules.py]] 的 `CTCHead(vocab_size=29)` 一致。

---

## 為什麼音訊這層不用處理變長

> [!success] Whisper 固定 pad 到 30 秒 → 每一句的 `input_features` 都是 `(80, 3000)`，完全等長
> 所以 `collate_fn` 對音訊直接 `torch.stack` 就好。**變長問題只剩文字那邊**（`ctc_ids`）。
>
> 壓縮（`ctc_compress`）發生在模型內部的 forward 裡，Dataset 看不到。

`ctc_ids` 的 padding：

```python
ctc_lens = torch.tensor([len(b["ctc_ids"]) for b in batch], dtype=torch.long)
max_len = int(ctc_lens.max())
ctc_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
for i, b in enumerate(batch):
    ctc_ids[i, : len(b["ctc_ids"])] = b["ctc_ids"]
```

`ctc_lens` 必須一路傳到 CTC loss — 見 [[modules.py]] 的陷阱 3。

---

## ⚠️ 必須過濾 duration > 30

Whisper 只處理前 30 秒，**超過的部分聲音消失但逐字稿完整** → 模型被要求輸出沒聽到的字，污染訓練。

全部 manifest 共 **38 句**（0.013%），dev-clean 有 9 句。直接過濾，記為 limitation。

```
data/manifest/dev-clean.tsv: 載入 2694 句，跳過 9 句（>30.0s）
```

---

## ⚠️ DataLoader 必須設 num_workers

```python
DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_fn,
           num_workers=8, pin_memory=True, drop_last=True)
```

`__getitem__` 每次要**讀 flac + 算 3000 frame 的 FFT**（純 CPU 重活），而 28 萬句 × 10 epoch = **280 萬次**。

| 參數 | 作用 |
|---|---|
| `num_workers=8` | 開 8 個子行程在背景**預先**讀檔並算特徵。GPU 算這批的同時 CPU 已在準備下一批。預設 0 = 主行程做完 I/O 才開始算，GPU 大半時間閒置 |
| `pin_memory=True` | 資料放「固定記憶體」，CPU→GPU 傳輸更快 |
| `drop_last=True` | 丟掉最後不滿一個 batch 的殘量，讓每步 batch size 一致 |

---

## 驗證

```bash
python src/dataset.py
```

```
data/manifest/dev-clean.tsv: 載入 2694 句，跳過 9 句（>30.0s）

單筆: 1272-128104-0000
  input_features (80, 3000)
  ctc_ids        (89,)
  text           mister quilter is the apostle of the middle classe...
  rare_words     ['apostle', 'gospel', 'quilter']

一個 batch:
  input_features (4, 80, 3000)
  ctc_ids        (4, 172)
  ctc_lens       [89, 63, 172, 133]
```

> [!success] 手算交叉驗證
> 第一句逐字稿 `mister quilter is the apostle of the middle classes and we are glad to welcome his gospel`：
> **73 個字母 + 16 個空格 = 89** → 吻合 `ctc_ids (89,)`
> 而 17 個單字也對上 [[parse_trans.py]] 當初算的 `n_words = 17` ✓

---

## 語法筆記

**用欄位名稱查索引，不要寫死位置**：

```python
header = f.readline().rstrip("\n").split("\t")
idx = {name: i for i, name in enumerate(header)}
dur = float(p[idx["duration"]])
```

這樣哪天 manifest 欄位順序變了也不會錯位。
