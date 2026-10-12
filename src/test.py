"""檢查訓練後 ctc_compress 是否學到對齊。

判準：壓縮後的 token 數 / 實際字元數 → 接近 1.0 代表 CTC 學到了。
訓練前的基準是 158/89 = 1.78。
"""

import torch
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor

from dataset import LibriSpeechDataset, collate_fn
from model import COALAModel, WHISPER_ID
from modules import ctc_compress

CKPT = "exp/stage1_tc100/best.pt"


def main():
    dev = torch.device("cuda")
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    model = COALAModel().to(dev)
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert len(unexpected) == 0, unexpected
    model.eval()
    print(f"載入 {CKPT}（epoch {ck['epoch']}, step {ck['step']}）\n")

    with torch.no_grad():
        h, logits = model.encode_audio(batch["input_features"].to(dev))

    print(f"{'utt_id':22}{'壓縮後':>8}{'實際字元':>10}{'比值':>8}")
    print("-" * 48)
    ratios = []
    for i, utt in enumerate(batch["utt_id"]):
        n = ctc_compress(h[i], logits[i]).shape[0]
        n_char = int(batch["ctc_lens"][i])
        r = n / n_char
        ratios.append(r)
        print(f"{utt:22}{n:>8}{n_char:>10}{r:>8.2f}")
    print("-" * 48)
    print(f"{'平均比值':22}{'':>8}{'':>10}{sum(ratios)/len(ratios):>8.2f}")
    print(f"\n訓練前的基準: 158/89 = 1.78")
    print(f"判準: 接近 1.0 = CTC 學到對齊；1.5 以上 = 沒學好")


if __name__ == "__main__":
    main()
    