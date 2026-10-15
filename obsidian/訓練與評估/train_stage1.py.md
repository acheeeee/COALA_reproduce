---
tags: [COALA, 訓練, 檔案]
層: 訓練與評估
狀態: 進行中
---

# train_stage1.py

← [[訓練與評估 總覽]] · [[COALA 索引]]

引用 [[model.py]]、[[dataset.py]]、[[biasing.py]]。

> [!todo] 狀態：正在跑 train-clean-100 的 10 epochs（約 8.3 小時）


<!-- 結構:start -->
## 🧩 結構

**整合出的功能**：Stage 1（ASR）完整訓練 —— 資料、模型、優化器、排程、驗證、tensorboard、epoch 摘要、checkpoint、續跑。

| 類型 | 內容 |
|---|---|
| import 外部 | `argparse`、`random`、`time`、`pathlib`、`torch`、`DataLoader`、`SummaryWriter`、`tqdm`、`transformers`（`WhisperFeatureExtractor`、`get_linear_schedule_with_warmup`） |
| import 自己的 | [[biasing.py]]、[[dataset.py]]、[[model.py]] |
| 被誰 import | 無 |

**定義的函式**

| 函式 | 一句話 |
|---|---|
| [[train_stage1.parse_args()]] | → [[args]] |
| [[make_entity_lists()]] | batch → [[entity_lists]] |
| [[to_device()]] | 三個 tensor 搬上 GPU |
| [[save_ckpt()]] | 只存可訓練參數 → [[checkpoint .pt]] |
| [[validate()]] | dev-clean 前 50 batch 的平均 loss |
| [[train_stage1.main()]] | 訓練迴圈 |

**主要物件**：[[args]]、[[DataLoader (dl)]]、[[rng]]、[[params]]、[[opt]]、[[sched]]、[[writer]]、[[batch]]、[[out (stage1 loss dict)]]

**產出的檔案**：[[checkpoint .pt]]、[[epoch_summary.tsv]]、[[tensorboard events]]
<!-- 結構:end -->

---

## 功能

Stage 1（ASR）主訓練腳本。

論文 3.4：**10 epochs、batch size 4、CTC 權重 0.3 + CE 1.0**。
論文**未給** lr / optimizer / scheduler（TODO 疑點 #1）→ 自行選定並記錄。

---

## 函式

| 函式 | 輸入 | 輸出 |
|---|---|---|
| `parse_args()` | 命令列 | `argparse.Namespace` |
| `make_entity_lists(rare_words_batch, rare_pool, rng)` | `list[list[str]]`、`list[str]`、`random.Random` | `list[list[str]]` — 每句 10 個 entity |
| `to_device(batch, dev)` | `dict`、`torch.device` | `dict` — 只搬 `input_features`/`ctc_ids`/`ctc_lens` |
| `save_ckpt(path, model, opt, sched, step, epoch, args)` | — | 無（寫檔） |
| `validate(model, val_dl, rare_pool, dev, max_batches=50)` | — | `(total, ctc, ce)` 平均 loss |
| `main()` | 無 | 無 |

---

## 命令列參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `--train` | `data/manifest/train-clean-100.tsv` | |
| `--val` | `data/manifest/dev-clean.tsv` | |
| `--out` | `exp/stage1_tc100` | checkpoint + tensorboard |
| `--epochs` | 10 | **論文設定** |
| `--batch-size` | 4 | **論文設定** |
| `--lr` | 1e-4 | 論文未給 |
| `--warmup-ratio` | 0.03 | 論文未給 |
| `--grad-clip` | 1.0 | 論文未給 |
| `--log-every` | 50 | |
| `--val-every` | 2000 | |
| `--seed` | 42 | 論文未給 |
| `--num-workers` | 8 | |
| `--resume` | None | checkpoint 路徑 |

---

## 三個自行決定的超參數（都要寫進 thesis）

| 設定 | 值 | 理由 |
|---|---|---|
| optimizer | AdamW, `weight_decay=0.01` | LoRA 微調的常見起點。過擬合測試用這個值收斂良好 |
| lr schedule | linear warmup **3%** + linear decay | adapter / CTC head 是隨機初始化，一開始梯度很大。warmup 讓 lr 從 0 慢慢爬，避免前幾百步把參數打壞 |
| grad clip | **1.0** | CTC loss 偶爾產生很大的梯度，裁剪範數防止單一異常 batch 毀掉訓練 |

---

## 訓練迴圈核心

```python
for batch in pbar:
    step += 1
    batch = to_device(batch, dev)
    el = make_entity_lists(batch["rare_words"], rare_pool, rng)

    opt.zero_grad()
    out = model.forward_stage1(batch, el)
    out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
    opt.step()
    sched.step()
```

`sched.step()` 要在 `opt.step()` **之後**。

---

## 輸出

```
exp/stage1_tc100/
├── best.pt                    val loss 最低的
├── epoch0.pt ~ epoch9.pt
└── events.out.tfevents.*      tensorboard
```

tensorboard 記錄：`train/total_loss`、`train/ctc_loss`、`train/ce_loss`、`train/lr`、`val/*`

---

## 啟動與監看

```bash
nvidia-smi                                 # 1. 先確認 GPU 1 沒人在用

export HF_HOME=/mnt/disk2/M11515125/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # ⚠️ 必須在同一個 shell export
export CUDA_VISIBLE_DEVICES=1              #    否則背景程序抓不到
python -c "import torch; print(torch.cuda.get_device_name(0))"   # 2. 確認是 3090

nohup python src/train_stage1.py > exp/stage1_tc100.log 2>&1 &
echo $!                                    # 3. 記下 PID

tail -f exp/stage1_tc100.log               # Ctrl-C 只停止看，不停訓練
tensorboard --logdir exp/ --port 6006
# 從本機連：ssh -L 6006:localhost:6006 bedo@79
```

### 怎麼判讀 `nvidia-smi` 有沒有人在用

| 看哪裡 | 空閒的樣子 | 有人在用 |
|---|---|---|
| `GPU-Util` | `0%` | `97%` |
| `Memory-Usage` | `12MiB / 24576MiB` | `18432MiB / 24576MiB` |
| `Processes` | 只有 `/usr/lib/xorg/Xorg`（顯示伺服器） | 出現 `python` + 幾 GB，類型 `C`(Compute) |

---

## 啟動後前 10 分鐘要確認的三件事

1. **速度 `~0.38 s/it`**（約 2.6 it/s）。如果是 1.04 s/it → bf16 沒生效
2. **loss 從 9.5 附近開始下降**
3. **總步數 `71,340`**（7134 步/epoch × 10 epochs）

---

## 實際執行紀錄（train-clean-100）

```
總步數 71,340 (7134 步/epoch × 10 epochs)
2.63 it/s = 0.38 s/it
step  111  loss 6.07
step 1415  loss 4.58   ce 3.70  ctc 2.94
ETA 約 50 分/epoch → 10 epochs 約 8.3 小時
```

---

## 為什麼先跑 train-clean-100 而不是 960h

| 資料集 | 步數/epoch | 單 epoch | 10 epochs |
|---|---|---|---|
| train-clean-100 (28539 句) | 7,134 | 50 分 | **8.3 小時** |
| train-960 (281241 句) | 70,310 | 7 小時 | **2.9 天** |

> [!note] 先用小的確認 WER 合理、loss 曲線正常，再上全量
> 有問題時損失 8 小時而不是 3 天。這也符合 downscale 的規劃 ——
> **論文用 960h，我先用 100h，這是必須報告的差異。**

---

## checkpoint 只存可訓練參數

```python
trainable = {k: v for k, v in model.state_dict().items()
             if k in {n for n, p in model.named_parameters() if p.requires_grad}}
```

只存 3.9M（約 15MB），不存凍結的 encoder/LM（省 2.5GB）。

> [!warning] 載入必須用 `strict=False`，但要檢查回傳值
> `load_state_dict` 預設要求鍵**完全吻合** → 會因缺少幾千個凍結參數的鍵而報錯。
>
> 但 **`strict=False` 會靜默忽略拼錯的鍵** → 若改過模組名稱，載入會「成功」但權重沒進去、**模型變回隨機初始化**。
>
> ```python
> missing, unexpected = model.load_state_dict(ck["model"], strict=False)
> assert len(unexpected) == 0, f"checkpoint 有不認識的鍵: {unexpected}"
> print(f"跳過 {len(missing)} 個凍結參數的鍵")
> ```
>
> `unexpected`（checkpoint 有但模型沒有的鍵）**必須為空**。

---

## `validate()` 的設計

```python
@torch.no_grad()
def validate(model, val_dl, rare_pool, dev, max_batches=50):
    model.eval()
    rng = random.Random(0)          # ⚠️ 固定 seed
    ...
    model.train()
    model.encoder.eval()            # ⚠️ 記得扳回來
```

- **固定 seed** → 每次驗證用同一組 biasing list，數字才可比較
- `max_batches=50` × batch 4 = 200 句，不用跑完整個 dev-clean
- 離開時要 `model.train()` **並且** `model.encoder.eval()`（見 [[overfit_test.py]]）
- 目前只算 loss，**WER 另外做**（需要 `generate()`，見 [[evaluate.py]]）

---

## 煙霧測試（不要直接跑 10 epochs）

```bash
python src/train_stage1.py --train data/manifest/dev-clean.tsv \
    --epochs 1 --val-every 200 --out exp/smoke
```

dev-clean 只有 2694 句，約 4~14 分鐘。確認：

1. loss 有在下降（從 ~9.5 開始）
2. `exp/smoke/` 裡有 `epoch0.pt` 和 `best.pt`
3. `tensorboard --logdir exp/` 看得到曲線

實際煙霧測試結果：`val loss 4.977 → 4.858 → 4.797` ✓

---

## 踩過的坑

> [!warning] `...` 是合法的 Python
> 留 TODO 用的 `...`（`Ellipsis` 物件）**不會報錯**，只會什麼都不做。
> 症狀：程式印完參數量就結束，訓練迴圈完全沒跑。
