---
tags: [COALA, 物件, 資料結構]
類別: 資料結構
型別: list[tuple[str, str]]
---
# items (trans 解析結果)

← [[物件 總覽]]

**存什麼**：一個 [[trans.txt]] 解析後的所有句子。

| 項目 | 內容 |
|---|---|
| 型別 | `list[tuple[str, str]]` |
| 長相 | `[("1272-128104-0000", "MISTER QUILTER IS ..."), ...]` → `[(utt_id, text), ...]` |
| 長度 | 一個 chapter 的句數 |
| 定義於 | [[parse_trans_file()]] 回傳 |

注意這裡的 text 還是**原始大寫**。

## 誰在用
- [[build_manifest()]]、[[figure1.main()]]：`for utt_id, text in parse_trans_file(f)`
- [[parse_trans.main()]]：示範印前 3 句
