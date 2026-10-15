---
tags: [COALA, 工具, 檔案, 驗證]
層: 工具
狀態: 完成
---

# check_shape.py

← [[工具 總覽]] · [[COALA 索引]]

**一次性執行。** 不被任何檔案引用。

---

## 功能

量測 `音訊 → Whisper encoder → LM embedding` 的 shape 鏈條，**確定 audio adapter 的輸入輸出規格**。

因為論文沒寫 adapter 架構（TODO 疑點 #5），要自己設計；設計之前必須知道兩端的規格。

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `pack_two_utterances(manifest_path)` | manifest 路徑 | `[(utt_id, path, duration), ...]` — **最短 + 最長各一句** |
| `main()` | 無 | 無（印 shape 鏈條） |

```python
def pack_two_utterances(manifest_path):
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_utt = header.index("utt_id")
        i_path = header.index("audio_path")
        i_duration = header.index("duration")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append((parts[i_utt], parts[i_path], float(parts[i_duration])))

    rows.sort(key=lambda r: r[2])      # 依 duration 排序
    return [rows[0], rows[-1]]         # 最短 + 最長
```

**刻意挑最短和最長兩句**，就是為了驗證「Whisper 一律 pad 到固定長度」。

> [!tip] 兩個語法點
> **`header.index("utt_id")` —— 用欄位名稱查索引，不要寫死 `parts[0]`。** 哪天 manifest 欄位順序變了也不會錯位。
>
> **`rows.sort(key=lambda r: r[2])`** —— `key=` 告訴 sort「拿每個元素的這個計算結果去比」。`lambda r: r[2]` 是不取名字的函式，意思是「給我 r，回傳 r[2]」（也就是 duration）。排序後 `rows[0]` 最短、`rows[-1]` 最長（`-1` 是倒數第一個）。
>
> 這裡用 `.sort()`（原地排序）是安全的，因為 `rows` 是函式內部自己建的 local 變數，改掉它不影響外面。**只有收到別人傳進來的 list 時才要避免 `.sort()`，改用 `sorted()`。**

---

## 實測結果

```
波形 (16000 Hz)  →  log-Mel (1, 80, 3000)  →  encoder (1, 1500, 1280)
```

| 測試音檔 | 時長 | 原始波形 | log-Mel | encoder 輸出 |
|---|---|---|---|---|
| 3081-166546-0073 | 1.445 s | (23120,) | (1, 80, 3000) | (1, 1500, 1280) |
| 422-122949-0013 | 32.645 s | (522320,) | (1, 80, 3000) | (1, 1500, 1280) |

> [!success] 兩句話的 shape 完全一樣
> 證實 **Whisper 固定 pad/截斷到 30 秒**（3000 frames = 30s × 100 fps），
> encoder 的 conv `stride=2` 把 3000 對折成 **T = 1500**。
>
> **Audio adapter 規格因此確定：輸入 `(B, 1500, 1280)` → 輸出 `(B, ?, 576)`**

LM embedding 查表 shape = `(49152, 576)` → adapter 的輸出維度必須是 **576**。

---

## 兩個關鍵推論

> [!important] 推論 1：adapter 必須降採樣，不只換維度
> SmolLM2 的 context window 只有 **2048**：
> ```
> 1500 audio + ~30 instruction + ~5 entity + ~40 transcription ≈ 1575
> ```
> 單次勉強可行，但 **Stage 2 一個 utterance 要跑 121 次 forward** → 計算上不可行。
> **這正是論文 Table 2「N=5000 → OOM」的機制。**
>
> → 論文 3.4 那句 *"CTC module generates frame-level alignments that are projected into the backbone LM embedding space as audio tokens"* 的真正意義：
> **用 CTC 預測結果濾掉 blank 與重複 frame**（CTC compression），把 1500 壓到與實際字數同量級（約數十個）。
>
> 1500 frames 對應約 40 個字 → 平均一字佔 **37 frames**，其餘全是 blank/重複，壓縮空間極大。
>
> 實作見 [[modules]] 的 `ctc_compress`。

> [!warning] 推論 2：>30 秒的音檔被靜默截斷
> 最長的 dev-clean 句子 32.645 秒 → Whisper 只處理前 30 秒，
> **後 2.6 秒聲音消失但逐字稿完整**，模型被要求輸出沒聽到的字。
>
> → [[dataset]] 用 `duration > 30` 過濾。全部 manifest 共 **38 句**（0.013%），影響可忽略但要記為 limitation。
>
> 統計方式：
> ```bash
> for f in data/manifest/*.tsv; do echo -n "$f: "; awk -F'\t' 'NR>1 && $3>30' "$f" | wc -l; done
> ```
> ⚠️ 用 `awk ... END {print c}` 配多個檔案時，`END` 只執行**一次** → 印出的是**總和**，不是單檔。

---

## 基礎概念筆記

### 音訊 → log-Mel 的壓縮

- 16000 Hz = 每秒 16000 個數字。3 秒 = 48000 個 → Transformer 的注意力矩陣會是 48000² = **23 億**，不可行
- log-Mel 的四步：
  1. 切 25ms 窗、每 **10ms** 移一格（窗口重疊）
  2. 每段做傅立葉轉換 → 各頻率的強度
  3. 頻率軸壓成 **80 個 Mel 頻帶**（模擬人耳對頻率的非線性感知：對低頻敏感、高頻不敏感）
  4. 取 **log**（人耳對音量也是對數感知）
- 結果 `(80, 時間格數)`，1 秒 = 100 格。從 16000 數字/秒 變成 8000 數字/秒且**結構化**

### Whisper encoder 做什麼

log-Mel 只是「聲音的物理描述」，不帶語意。encoder（32 層 Transformer）把它轉成**帶語意的向量** —— 讓「發 /kæt/ 這個音」變成能代表「cat 的發音」的向量。

輸出 `(1500, 1280)`：每個時間格對應一個 1280 維向量，可想成「這 20 毫秒的聲音，用 1280 個數字描述它的語意」。

### 「audio token」為什麼能存在

```
文字流程：  "cat" → tokenizer → id 3421 → embedding 查表 (49152,576) 的第 3421 列 → 576 維向量
```

> [!important] 關鍵洞察
> LM 的 Transformer 層其實**不知道「token」這回事**，它只看到一串 **576 維向量**。
> token id 和查表只是**產生那些向量的手段**。
>
> 所以可以自己塞任何 576 維向量進去 —— 這就是 SLM（speech-augmented LM）的核心把戲：
> ```
> Whisper encoder (T, 1280)
>     ↓ audio adapter：1280 → 576
> (T, 576)  ← 這些向量不對應詞彙表裡任何詞，但 LM 吃得下
>     ↓ 接在 instruction 和 entity 的 embedding 前面
> LM 看到：[T 個音訊向量][instruction 向量][entity 向量]
> ```
>
> 論文圖 (a) 的 **Audio Tokens** 就是這個。它們**佔據 token 的位置，但不對應詞彙表裡任何詞**。
>
> 實作見 [[prompt]] 的 `inputs_embeds` 路徑。

---

## 踩過的坑

> [!warning] `py_compile` 通過不代表程式會跑完
> 第一次跑時第 21 行有 `manifest_pate` 這個打字錯誤 → `NameError`，但它在**執行到那一行才會發現**，所以只印出第一組 shape 就中斷。
>
> ```bash
> python -m py_compile src/*.py     # 只抓語法錯誤
> pip install pyflakes
> pyflakes src/*.py                 # 不執行就能抓「變數沒定義」「import 沒用到」
> ```
>
> `pyflakes` 在目前階段投報率很高 —— 多數錯誤是打字和命名，一秒就能掃出來，不用等程式跑到那一行。
