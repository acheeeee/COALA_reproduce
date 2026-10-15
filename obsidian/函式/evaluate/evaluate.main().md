---
tags: [COALA, 函式, 評估]
種類: main
檔案: evaluate.py
---
# evaluate.main()

← [[evaluate.py]] · [[函式 總覽]]

**功能**：載入 checkpoint，對整個 manifest 逐句生成，寫出 [[hyp TSV]]，再交給 [[score.py (官方)]] 算 WER。

## 步驟
1. [[evaluate.parse_args()]] → [[args]]
2. [[fe]] → [[LibriSpeechDataset]]`(args.manifest, fe)` ⚠️ Bug B：沒傳 `max_duration=1e9`
3. `COALAModel(stack=args.stack)` → 載入 [[ck (checkpoint dict)]]（`strict=False`，assert 無 unexpected）
4. `model.eval()`、`model.lm.set_adapter("asr")`
5. [[load_rare_pool()]]、`Random(args.seed)`
6. 逐句（不經 DataLoader）：
   - 依 `--mode` 決定 [[entity_list]]：`[]` / [[positives (E⁺)]] / [[make_stage1_list()]]
   - [[transcribe_one()]] → [[hyp]]
   - 寫一行 `utt_id \t hyp`
   - 前 3 句印 REF / HYP 對照

## 用到的物件
[[args]]、[[fe]]、[[sample (Dataset 單筆)]]、[[ck (checkpoint dict)]]、[[rare_pool]]、[[rng]]、[[entity_list]]、[[hyp]]、[[hyp TSV]]

## 兩個 bug（STATUS 待辦 #2）

| Bug | 位置 | 症狀 | 修法 |
|---|---|---|---|
| A | [[transcribe_one()]] | 2629 行 vs 2611 句 | `" ".join(text.split())` |
| B | 這裡第 2 步 | 少測 9 句，score.py 報 missing | `LibriSpeechDataset(args.manifest, fe, max_duration=1e9)` |

> 逐句生成很慢（test-clean 約 2 小時）。可以兩張卡各跑一組設定並行。
