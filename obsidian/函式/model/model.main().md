---
tags: [COALA, 函式, 模型層]
種類: main
檔案: model.py
---
# model.main()

← [[model.py]] · [[函式 總覽]]

**功能**：整個 Stage 1 模型的冒煙測試 —— 參數量、三個 loss 的量級、**梯度是否到達每一個可訓練張量**。

## 步驟
1. [[dev]]：有 GPU 用 cuda，否則 cpu
2. 資料：[[fe]] → [[LibriSpeechDataset]]（dev-clean）→ DataLoader（batch 2）→ 一個 [[batch]]
3. [[load_rare_pool()]] + `Random(42)` → 每句 [[make_stage1_list()]] → [[entity_lists]]
4. `COALAModel().to(dev)`，印可訓練參數量
5. 三個 tensor 欄位搬上 GPU → [[COALAModel.forward_stage1()]] → 印三個 loss
6. `out["loss"].backward()` → 數有梯度的張量 / 可訓練張量

## 判斷對不對

| 項目 | 預期 | 意義 |
|---|---|---|
| 可訓練參數 | 2,413,469（stack=2）；3,888,029（stack=4） | 架構和預算推算吻合 |
| ce_loss | 遠低於 ln(49152) = 10.80（實測 5.60） | 預訓練權重有載入 |
| ctc_loss | 約 10–15 | 隨機初始化的合理範圍 |
| 有梯度的張量 | **124/124** | 梯度穿過 argmax 選擇、cat、pad，沒有斷 |

> ⚠️ 第 4 步 `COALAModel()` 用預設 stack=2；早期筆記的 3,888,029 是 stack=4 時的數字。
