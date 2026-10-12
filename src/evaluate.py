"""產生 ASR 預測並用官方 score.py 算 WER / U-WER / B-WER。

論文 Table 2 的 COALA 列（B-WER (WER / U-WER)）：
    w/o Biasing   test-clean 23.39 (4.54 / 2.34)   test-other 39.49 (8.75 / 5.48)
    Oracle        test-clean  1.77 (1.90 / 1.92)   test-other  6.12 (4.86 / 4.73)
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from biasing import load_rare_pool, make_stage1_list
from dataset import LibriSpeechDataset
from model import COALAModel, WHISPER_ID
from modules import ctc_compress
from prompt import build_gen_prompt

MAX_NEW_TOKENS = 200


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="data/manifest/dev-clean.tsv")
    ap.add_argument("--out", required=True, help="輸出的 hyp 檔路徑")
    ap.add_argument("--mode", default="nobias",
                    choices=["nobias", "oracle", "stage1"],
                    help="nobias=不給entity / oracle=只給正確答案 / stage1=10個entity")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 句（快速測試）")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def transcribe_one(model, input_features, entity_list, dev):
    """單句生成，回傳文字。"""
    feats = input_features.unsqueeze(0).to(dev)         # (1, 80, 3000)
    h, ctc_logits = model.encode_audio(feats)
    audio_tok = ctc_compress(h[0], ctc_logits[0])

    prompt = build_gen_prompt(audio_tok, entity_list,
                              model.tokenizer, model.lm.get_input_embeddings())

    # TODO 5: out_ids = model.lm.generate(
    #             inputs_embeds=prompt["inputs_embeds"],
    #             attention_mask=prompt["attention_mask"],
    #             max_new_tokens=MAX_NEW_TOKENS,
    #             do_sample=False,              # greedy，可重現
    #             num_beams=1,
    #             eos_token_id=model.tokenizer.eos_token_id,
    #             pad_token_id=model.tokenizer.eos_token_id,
    #         )
    out_ids = model.lm.generate(
                inputs_embeds=prompt["inputs_embeds"],
                attention_mask=prompt["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,              # greedy，可重現
                num_beams=1,
                eos_token_id=model.tokenizer.eos_token_id,
                pad_token_id=model.tokenizer.eos_token_id,
            )
    # TODO 6: text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    #         return text.strip()
    text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


def main():
    args = parse_args()
    dev = torch.device("cuda")

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset(args.manifest, fe)

    model = COALAModel(stack=args.stack).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert len(unexpected) == 0, f"checkpoint 有不認識的鍵: {unexpected}"
    print(f"載入 {args.ckpt}（epoch {ck['epoch']}, step {ck['step']}），"
          f"跳過 {len(missing)} 個凍結參數")
    model.eval()
    model.lm.set_adapter("asr")

    rare_pool = load_rare_pool()
    rng = random.Random(args.seed)

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=args.mode):
            s = ds[i]
            if args.mode == "nobias":
                el = []
            elif args.mode == "oracle":
                el = s["rare_words"]
            else:
                el = make_stage1_list(s["rare_words"], rare_pool, rng)

            hyp = transcribe_one(model, s["input_features"], el, dev)
            f.write(f"{s['utt_id']}\t{hyp}\n")

            if i < 3:
                print(f"\n  REF: {s['text'][:80]}")
                print(f"  HYP: {hyp[:80]}")

    print(f"\n寫入 {out_path}（{n} 句）")


if __name__ == "__main__":
    main()