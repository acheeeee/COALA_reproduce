"""只用 Stage 1 的 CTC head 轉寫（不經過 LM），看 LM 拿到的音訊 token 裡還剩多少 rare word 資訊。

ctc_compress 只留下 CTC argmax 不是 blank、且和前一格不同的 frame，
所以 LM 看到的音訊 token 一個對一個就是 CTC 預測的字元。
CTC 把 rare word 拼錯時，LM 只能從拼錯的那幾格去猜；給 Oracle 清單時才能救回來。

用法：
    python src/ctc_decode.py --ckpt exp/stage1_960/best.pt --manifest data/manifest/test-clean.tsv \
        --out exp/hyp/960h.test-clean.ctc.tsv
    → 再用 score_wer.py 算 B-WER，和 w/o（LM 解碼）比較
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from dataset import CTC_BLANK_ID, CTC_CHARS, LibriSpeechDataset
from eval_utils import AMP_CHOICES, load_stage1_model
from model import WHISPER_ID


def ctc_greedy(logits):
    """(L, 29) → 文字：argmax、合併連續重複、去掉 blank（和 ctc_compress 留下的格子一致）。"""
    pred = logits.argmax(dim=-1).tolist()
    out, prev = [], None
    for p in pred:
        if p != prev and p != CTC_BLANK_ID:
            out.append(CTC_CHARS[p - 1])
        prev = p
    return " ".join("".join(out).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest/test-clean.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--amp", default="auto", choices=AMP_CHOICES)
    args = ap.parse_args()

    dev = torch.device("cuda")
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset(args.manifest, fe, max_duration=1e9)
    model = load_stage1_model(args.ckpt, args.stack, dev, args.amp)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tok = n_char = 0
    with open(out_path, "w", encoding="utf-8") as f, torch.no_grad():
        for i in tqdm(range(n), desc="ctc", mininterval=30):
            s = ds[i]
            _, logits = model.encode_audio(s["input_features"].unsqueeze(0).to(dev))
            hyp = ctc_greedy(logits[0])
            f.write(f"{s['utt_id']}\t{hyp}\n")
            pred = logits[0].argmax(-1)
            n_tok += int(((pred != CTC_BLANK_ID) & torch.cat([pred[:1] + 1, pred[:-1]]).ne(pred)).sum())
            n_char += len(s["text"])
            if i < 3:
                print(f"\n  REF: {s['text'][:80]}\n  CTC: {hyp[:80]}", flush=True)

    print(f"\n寫入 {out_path}（{n} 句）")
    print(f"LM 拿到的音訊 token 平均每句 {n_tok / n:.1f} 個；參考答案平均 {n_char / n:.1f} 個字元（含空格）")


if __name__ == "__main__":
    main()
