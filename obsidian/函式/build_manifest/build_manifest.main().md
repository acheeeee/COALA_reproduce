---
tags: [COALA, 函式, 資料層]
種類: main
檔案: build_manifest.py
---
# build_manifest.main()

← [[build_manifest.py]] · [[函式 總覽]]

**功能**：命令列入口，一次為多個 split 建 manifest 並印統計。

```bash
python src/build_manifest.py dev-clean test-clean train-clean-100
```

## 步驟
1. argparse：`splits`（`nargs="+"` = 一個以上的位置參數）
2. [[load_common_words()]] → [[common_set]]（只載入一次，所有 split 共用）
3. 每個 split：
   - [[build_manifest()]] → `rows`
   - [[write_manifest()]] → [[MANIFEST_DIR]] `/ f"{split}.tsv"`
   - 統計：總時數、無 rare word 的比例（`n_zero`）、多目標（≥2 個 rare word）的比例（`n_multi`）

## 用到的物件
[[common_set]]、[[row (manifest 一列)]]、[[MANIFEST_DIR]]、[[manifest TSV]]

統計時要 `json.loads(r["rare_words"])` 把 JSON 字串轉回 list 才能算長度。
