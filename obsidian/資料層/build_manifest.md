---
tags: [COALA, 資料層, 檔案]
層: 資料層
狀態: 完成
---

# build_manifest.py

← [[資料層 總覽]] · [[COALA 索引]]

**一次性執行。** 引用 [[parse_trans]] 和 [[rare_words]]。產出給 [[dataset]] 讀。

---

## 功能

把 LibriSpeech 的散落檔案整理成一張表，一行一句，並算好每句的 E⁺。

**輸入**：`LibriSpeech/<split>/<speaker>/<chapter>/` 底下的 flac + trans.txt
**輸出**：`data/manifest/<split>.tsv`

之後所有程式只讀 manifest，不再爬 28 萬個目錄。

```bash
python src/build_manifest.py dev-clean dev-other test-clean test-other
python src/build_manifest.py train-clean-100 train-clean-360 train-other-500
```

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `build_manifest(split, common_set)` | `split: str` 例 `"dev-clean"`、`common_set: set` | `list[dict]` — 每個 dict 有 6 個鍵 |
| `write_manifest(rows, out_path)` | `rows: list[dict]`、`out_path: Path` | 無（寫檔） |
| `main()` | argparse 的 `splits`（`nargs="+"`，可多個） | 無（寫檔 + 印統計） |

常數：`LIBRISPEECH_ROOT`、`MANIFEST_DIR`、`FIELDS`

---

## 輸出欄位（TSV，tab 分隔，第一行標頭）

| 欄位 | 範例 | 用途 |
|---|---|---|
| `utt_id` | `1272-128104-0000` | 與官方 ref 檔對齊的鍵 |
| `audio_path` | `data/LIBRISPEECH/.../xxx.flac` | 由命名規則推出 |
| `duration` | `13.375` | 過濾 >30s、按長度排序、算總時數 |
| `n_words` | `17` | 統計用 |
| `text` | `mister quilter is...` | **已轉小寫** |
| `rare_words` | `["apostle","gospel","quilter"]` | **= E⁺**，JSON 陣列 |

---

## 八個步驟（一句話走完全程）

**步驟 0 — 載入 common word set**（迴圈外，只做一次）

**步驟 1 — `rglob` 找出所有 trans.txt**

```python
trans_files = sorted(split_dir.rglob("*.trans.txt"))     # dev-clean 找到 97 個
```

`glob` 只找當前目錄，`rglob`（**r**ecursive）會往下鑽所有子目錄。結構是 `split/speaker/chapter/` 兩層深，必須用 rglob。

外面包 `sorted()` 是為了**固定順序** — 檔案系統回傳順序不保證一致，排序過才能保證今天和三個月後跑出來逐行相同。

**步驟 2 — 解析**（呼叫 [[parse_trans]]，巢狀迴圈：97 個檔案 × 每檔十幾句 = 2703 句）

**步驟 3 — 從 utt_id 推出音檔路徑**

```python
audio_path = trans_file.parent / f"{utt_id}.flac"
if not audio_path.is_file():
    raise FileNotFoundError(f"找不到音檔: {audio_path}")
```

trans.txt 裡**只有文字，沒有音檔路徑**。靠 LibriSpeech 的命名規則：同一個資料夾裡的同名 flac。`.parent` 取「這個檔案所在的資料夾」。

> [!tip] 為什麼要故意讓程式崩掉
> 若有音檔缺失，不檢查的話會安靜地寫出一份少了幾百句的 manifest，然後訓練到一半莫名出錯。
> **與其安靜地做錯，不如大聲地停下來。**

**步驟 4 — 讀檔頭取時長**

```python
"duration": round(sf.info(str(audio_path)).duration, 3)
```

`sf.info()` **只讀檔頭**（取樣率 + 總樣本數），不解碼音訊。這是效能關鍵：

| 做法 | 單檔成本 | 28 萬檔 |
|---|---|---|
| 只讀檔頭 | < 1 ms | 十幾分鐘 |
| 真讀波形 | 幾十 ms | 好幾小時 |

**步驟 5 — 轉小寫**

```python
text = text.lower()
```

> [!important] 為什麼在「這裡」轉
> manifest 裡存的已經是小寫，後面**所有**程式（訓練、評估、biasing）都只讀 manifest，永遠碰不到大寫。
>
> 若不這樣做，就得在每支用到文字的程式裡都記得寫 `.lower()`。十支有一支忘記，就是一個找不到原因的 bug（E⁺ 永遠空、Recall 為 0、不報錯）。
>
> **把正規化固定在資料入口，而不是散落在每個使用處。**

**步驟 6 — 算 E⁺**（呼叫 [[rare_words]] 的 `find_rare_words`）

```python
"rare_words": json.dumps(find_rare_words(text, common_set))
```

`json.dumps` 把 list 轉成字串，因為 TSV 一個欄位只能放一個字串。不自己用逗號拼，是因為 rare word 含撇號（`abdi's`）等特殊字元，JSON 有定義好的轉義規則，而且官方 `ref/*.tsv` 用的也是 JSON，格式一致。

**步驟 7 — 組成 dict**（一句 = 一列。list of dict 就是「表格」在 Python 裡最自然的表示）

**步驟 8 — 按 `FIELDS` 順序寫 TSV**

```python
f.write("\t".join(FIELDS) + "\n")                        # 標頭
for r in rows:
    f.write("\t".join(str(r[k]) for k in FIELDS) + "\n")
```

用 `FIELDS` 常數控制順序很重要：**寫入和讀取參照同一份定義**，否則欄位錯位且不報錯。

---

## 踩過的坑

> [!warning] dict 的鍵名 ≠ 變數名
> 把詞表變數從 `rare_set` 改成 `common_set` 時，連 dict 的鍵名也一起改了：
> ```python
> "common_set": json.dumps(find_rare_words(text, common_set)),
> # ↑ 這是「輸出欄位名稱」，必須跟 FIELDS 一致     ↑ 這才是要改的
> ```
> → `write_manifest` 去拿 `r["rare_words"]` 找不到 → `KeyError`
>
> **冒號左邊是輸出欄位名，右邊才是輸入資料。兩者無關。**

> [!warning] 改完先 compile，不要直接跑大資料集
> ```bash
> python -m py_compile src/build_manifest.py    # 語法（0.1 秒）
> pyflakes src/build_manifest.py                # 未定義變數、未使用的 import
> ```
> `build_manifest.py` 跑 train-360 要十幾分鐘，如果錯誤在最後一行 print（例如打成 `nzero`），你要等十幾分鐘才知道。

---

## 驗證結果

| split | 句數 | 時長 | 平均句長 | 無 rare word | 多目標(2+) |
|---|---|---|---|---|---|
| dev-clean | 2703 | 5.39 h | 7.2 s | 23.68% | 48.91% |
| dev-other | 2864 | 5.12 h | 6.4 s | 29.71% | 41.93% |
| test-clean | 2620 | 5.40 h | 7.4 s | 24.43% | 49.20% |
| test-other | 2939 | 5.34 h | 6.5 s | 27.15% | 44.37% |
| train-clean-100 | 28539 | 100.59 h | 12.7 s | 7.42% | **78.43%** |
| train-clean-360 | 104014 | 363.61 h | 12.6 s | 7.35% | **78.77%** |
| train-other-500 | 148688 | 496.86 h | 12.0 s | 10.01% | **73.29%** |
| **訓練合計** | **281241** | **961.06 h** | | | |

全部精準吻合官方句數，總時長對上論文的 960h。

> [!success] 重要發現：訓練集的多目標比例遠高於評估集
> 訓練集 **73~79%** 的句子含 2 個以上 rare word，dev/test 只有 42~49%。原因是訓練句平均長 1.75 倍（12.6s vs 7.2s）。
>
> → **比論文 Figure 1 更強的動機支持**：訓練時有將近八成資料是多目標，全部受 discriminative loss 的 inter-positive competition 影響。這解釋了為何 baseline Bias-Loss 需要 auxiliary log loss（w=0.1）才能收斂，而 MPD/DPD-Loss 能獨立收斂。

---

## 為什麼要有 manifest 這道中間層

| 不用 manifest | 用 manifest |
|---|---|
| 每個 epoch 重爬目錄、重算 rare word | 算一次、重複用 |
| 資料不可檢查 | `head`、`wc -l`、`awk` 都能直接看 |
| 「我當初大概是這樣跑的」 | 一個**具體檔案**，可標日期、可進 git（純文字，dev-clean 才 600KB） |

```bash
head -3 data/manifest/dev-clean.tsv | cut -f1-4 | column -t -s $'\t'
awk -F'\t' 'NR>1 && $3 > 25' data/manifest/dev-clean.tsv | wc -l   # 超過 25 秒的有幾句
```

幾乎所有正式 ASR 專案都有這一層，只是名字不同 — manifest、scp、data.json。
