---
tags: [COALA, 函式, 訓練]
種類: main
檔案: overfit_test.py
---
# overfit_test.main()

← [[overfit_test.py]] · [[函式 總覽]]

**功能**：同一個 batch（2 句）訓練 [[STEPS]] = 200 步，loss 應該降到接近 0。降不下來 = 實作有 bug。

## 步驟
1. `torch.manual_seed(0)`
2. dev-clean 取第一個 [[batch]]（batch 2），三個 tensor 搬上 GPU，**之後重複使用**
3. [[load_rare_pool()]] + `Random(42)` → [[entity_lists]]（**也固定**，不每步重抽）
4. [[COALAModel]]，`model.train()` + `model.encoder.eval()`
5. [[params]] → AdamW（lr = [[LR]]）
6. 200 步：`zero_grad` → [[COALAModel.forward_stage1()]] → `backward` → `step`；第 1 步和每 20 步印三個 loss

## 為什麼這樣能驗證架構
模型只要死記 2 句話。資料、entity 都固定 → 唯一能讓 loss 降不下來的原因是程式錯（梯度斷了、labels 對錯位置、凍結錯參數…）。

## 用到的物件
[[batch]]、[[entity_lists]]、[[params]]、[[opt]]、[[out (stage1 loss dict)]]
