"""過擬合測試：用同一個 batch 訓練 200 步，確認 loss 能降到接近 0。

這是驗證架構正確性的標準手法。模型只要死記 2 句話，
如果 loss 降不下來，就是實作有 bug，不是資料或超參數的問題。
"""

import random

import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor

from biasing import load_rare_pool, make_stage1_list
from dataset import LibriSpeechDataset, collate_fn
from model import COALAModel, WHISPER_ID

STEPS = 200
LR = 1e-4          # 論文未給（疑點 #1），先用 LoRA 常見值


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)

    # --- 固定一個 batch，之後重複用 ---
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))
    for k in ["input_features", "ctc_ids", "ctc_lens"]:
        batch[k] = batch[k].to(dev)

    rare_pool = load_rare_pool()
    rng = random.Random(42)
    entity_lists = [make_stage1_list(rw, rare_pool, rng)
                    for rw in batch["rare_words"]]

    # --- 模型與優化器 ---
    model = COALAModel().to(dev)
    model.train()                      # 訓練模式（但 encoder 保持 eval）
    model.encoder.eval()               # 凍結的 encoder 不要開 dropout

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    print(f"優化 {sum(p.numel() for p in params):,} 個參數，lr={LR}")

    # TODO: 寫訓練迴圈
    #   for step in range(1, STEPS + 1):
    #       1. opt.zero_grad()
    #       2. out = model.forward_stage1(batch, entity_lists)
    #       3. out["loss"].backward()
    #       4. opt.step()
    #       5. 每 20 步印一次 step / total / ctc / ce
    #       5. 每 20 步印一次 step / total / ctc / ce
    for step in range(1, STEPS + 1):
        opt.zero_grad() #empty the grad
        out = model.forward_stage1(batch, entity_lists)
        out["loss"].backward()
        opt.step()
        if step % 20 == 0 or step == 1:
                print(f"step {step:4d}  total {out['loss'].item():7.4f}  "
                f"ctc {out['ctc_loss'].item():7.4f}  "
                f"ce {out['ce_loss'].item():7.4f}")

if __name__ == "__main__":
    main()