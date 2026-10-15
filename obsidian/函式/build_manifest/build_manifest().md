---
tags: [COALA, 函式, 資料層]
種類: 函式
檔案: build_manifest.py
---
# build_manifest()

← [[build_manifest.py]] · [[函式 總覽]]

**功能**：走訪一個 LibriSpeech split 的所有句子，每句產生一列後設資料。

```python
def build_manifest(split, common_set):
```

## 輸入 → 輸出

| | 名稱 | 型別 | 物件 |
|---|---|---|---|
| 輸入 | `split` | `str` | 例 `"dev-clean"` |
| 輸入 | `common_set` | `set[str]` | [[common_set]] |
| 輸出 | `rows` | `list[dict]` | [[row (manifest 一列)]] 的清單（版本 1） |

## 內部物件

| 名稱 | 型別 | 角色 |
|---|---|---|
| `split_dir` | `Path` | [[LIBRISPEECH_ROOT]] `/ split`，不存在就 raise |
| `trans_files` | `list[Path]` | `sorted(split_dir.rglob("*.trans.txt"))` |
| `audio_path` | `Path` | `trans_file.parent / f"{utt_id}.flac"`，不存在就 raise |
| `text` | `str` | 轉小寫後的 [[text]] |

## 每一列怎麼算

| 欄 | 計算 |
|---|---|
| `duration` | `sf.info(audio_path).duration`（只讀檔頭，不解碼整個音檔 → 快），round 3 位 |
| `n_words` | `len(text.split())` |
| `rare_words` | `json.dumps(find_rare_words(text, common_set))` → **JSON 字串** |

## 呼叫了誰
[[parse_trans_file()]]、[[find_rare_words()]]

## 被誰呼叫
[[build_manifest.main()]]

> [!tip] 找不到檔案就直接 raise
> 寧可當場報錯，也不要默默跳過 → 不會產出「少了幾句但看不出來」的 manifest。
