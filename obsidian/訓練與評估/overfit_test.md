---
tags: [COALA, 訓練, 檔案, 驗證]
層: 訓練與評估
狀態: 完成
---

# overfit_test.py

← [[訓練與評估 總覽]] · [[COALA 索引]]

引用 [[model]]、[[dataset]]、[[biasing]]。

---

## 功能

**架構正確性驗證。** 拿同一個 batch（2 句）重複訓練 200 步，確認 loss 能降到接近 0。

常數：`STEPS = 200`、`LR = 1e-4`

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `main()` | 無 | 無（印 loss 曲線） |

---

## 為什麼這是最高投報率的檢查

> [!important] 在跑 960 小時之前，要先證明這個模型「學得會」
> 模型只要**死記**那兩句就好，容量絕對夠（135M 參數記 2 句話）。
>
> **如果 loss 掉不下來，就是實作有 bug**，不是資料不夠或要再訓練久一點。
>
> 可能的原因：梯度沒接通、learning rate 太小、labels 對錯位置、某個模組實際上沒被用到。
>
> 這些問題在完整訓練裡會表現成「loss 降得很慢」，你會以為是超參數問題，然後**浪費好幾天**。這裡五分鐘就能排除。

---

## 結果

```
step    1  total  9.5657  ctc 12.7997  ce  5.7258
step   20  total  1.8675  ctc  3.2897  ce  0.8806
step   40  total  1.4961  ctc  2.9687  ce  0.6055
step   60  total  0.8839  ctc  2.6797  ce  0.0800
step   80  total  0.7342  ctc  2.3890  ce  0.0174
step  100  total  0.6538  ctc  2.1121  ce  0.0202
step  120  total  0.5615  ctc  1.8559  ce  0.0047
step  140  total  0.4899  ctc  1.6216  ce  0.0034
step  160  total  0.4253  ctc  1.4118  ce  0.0017
step  180  total  0.3703  ctc  1.2293  ce  0.0015
step  200  total  0.3240  ctc  1.0752  ce  0.0015
```

> [!success] 判定 ✅
> `ce 5.73 → 0.0015`（完全死記 2 句）證明**梯度路徑、labels 對位、loss 權重全部正確**。
>
> `ctc 12.80 → 1.075` 且仍在下降 —— CTC 降得慢屬正常：adapter + CTC head 是**從零訓練**，而 CE 那側有預訓練 LM 幫忙。

（step 200 的 `ce` 從 0.0017 回彈到 0.0116 之類的微小震盪是正常的，不用理。）

---

## 判讀標準表

| 現象 | 判斷 |
|---|---|
| `ce` 降到 **< 0.1** | ✅ 架構正確，可以寫完整訓練 |
| `ctc` 降到 **< 1** | ✅ CTC 也在學 |
| 平緩下降但停在 2~3 | ⚠️ lr 太小，試 3e-4 或 1e-3 |
| 完全不動 | ❌ 梯度沒接通或 labels 錯位 |
| 變成 `nan` | ❌ lr 太大，或 CTC 的 inf 沒處理好 |

---

## ⚠️ 這個測試證明不了的事

> [!warning] `ce_loss → 0` 不保證模型在用音訊
> prompt 裡的 entity 清單**本身就含部分 rare word**，而且同一個 batch 重複 200 次 —— LM 完全可以學成「看到這組 entity 清單就背出這句話」，audio token 一眼都不看。
>
> 所以過擬合測試只證明「**架構可微分、能學**」，不證明「**音訊路徑有被使用**」。
>
> **真正能回答這件事的是 dev-clean 上的 WER**（模型沒見過的音訊）→ 見 [[evaluate]] 的 `--mode nobias`。
>
> 另一個直接檢查：過擬合後把 audio token 歸零，看 `ce_loss` 是否跳高。若不變 → 音訊沒被用到。

---

## 訓練三步式（之後所有訓練迴圈的骨架）

```python
for step in range(1, STEPS + 1):
    opt.zero_grad()                  # 清掉上一步殘留的梯度
    out = model.forward_stage1(batch, entity_lists)
    out["loss"].backward()           # 反向傳播，算出每個參數的梯度
    opt.step()                       # 用梯度更新參數

    if step % 20 == 0 or step == 1:
        print(f"step {step:4d}  total {out['loss'].item():7.4f}  "
              f"ctc {out['ctc_loss'].item():7.4f}  "
              f"ce {out['ce_loss'].item():7.4f}")
```

> [!danger] `zero_grad()` 不能忘
> PyTorch 的梯度是**累加**的。不清掉的話第 10 步會帶著前 9 步的梯度，更新量越來越大然後爆掉。

`.item()` 把單元素 tensor 轉成 Python float。直接印 tensor 會看到 `tensor(0.0015, device='cuda:0', grad_fn=...)` 很難讀，而且**把 tensor 存進 list 累加會讓整個計算圖留在記憶體裡不被釋放**，累積幾千步就 OOM。記 loss 一律用 `.item()`。

---

## ⚠️ 凍結模組的 train/eval 模式

```python
model.train()
model.encoder.eval()        # ← 必須單獨扳回來
```

> [!danger] `model.train()` 會遞迴套用到所有子模組
> 包含凍結的 Whisper encoder → **啟用 dropout** → 同樣的音訊，每一步算出**不同的** encoder 輸出。
>
> 後果：adapter 在學一個會抖動的目標，收斂變慢，而且**過擬合測試降不到 0**（因為輸入本身就在變）。
>
> **通則：每次 `.train()` 之後要把凍結的子模組設回 `.eval()`。**
>
> 更穩健的做法是在 model class 裡覆寫 `train()` 自動處理，但手動呼叫就夠。

---

## 執行

```bash
python src/overfit_test.py
```

約 1 分鐘（bf16 後）。改動任何模型層的 code 之後都該重跑一次，確認沒把學習能力弄壞。

**bf16 改動前後對照**（驗證 bf16 沒影響學習）：

| | fp32 | bf16 |
|---|---|---|
| step 200 `ce` | 0.0017 | **0.0015** |
| step 200 `ctc` | 1.134 | **1.075** |

幾乎相同（甚至略好）→ bf16 可安心使用。詳見 [[test]]。
