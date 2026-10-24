"""產生 ASR 預測並用官方 score.py 算 WER / U-WER / B-WER。

論文 Table 2 的 COALA 列（B-WER (WER / U-WER)）：
    w/o Biasing   test-clean 23.39 (4.54 / 2.34)   test-other 39.49 (8.75 / 5.48)
    Oracle        test-clean  1.77 (1.90 / 1.92)   test-other  6.12 (4.86 / 4.73)

--mode
    nobias  不給任何 entity
    oracle  只給正確答案（E⁺）
    stage1  和 Stage 1 訓練時相同：10 個 entity，最多 5 個正樣本
    file    從 --entities 指定的 JSON 讀每句的 entity 清單
            （bti.py 的輸出，或 score_stage2.py --dump-lists 的完整清單 → 重現 N=5000 OOM）

--decode
    greedy    每步取機率最高的 token（預設，可重現）
    official  官方 code（recipes/inference.py 第 312–322 行）的設定：
              beam 20 + sampling（temperature 0.8、top-k 20、top-p 0.5）、repetition penalty 1.1、
              no-repeat 3-gram、length penalty 0.5、最多 300 token；用 --seed 固定亂數
    greedy_norep  greedy 再加官方的兩個防重複設定（repetition penalty 1.1、no-repeat 3-gram），
              速度和 greedy 差不多。用來看整份清單放進 prompt 時的重複迴圈是不是解碼造成的。
              只傳 inputs_embeds 時，這兩個設定只看新生成的 token，不會懲罰 prompt 裡的 entity
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
from eval_utils import AMP_CHOICES, load_stage1_model
from model import WHISPER_ID
from modules import ctc_compress
from prompt import build_gen_prompt

MAX_NEW_TOKENS = 200

DECODE = {
    "greedy": dict(do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS),
    "official": dict(num_beams=20, do_sample=True, temperature=0.8, top_k=20, top_p=0.5,
                     repetition_penalty=1.1, no_repeat_ngram_size=3, length_penalty=0.5,
                     min_length=1, max_new_tokens=300),
    "greedy_norep": dict(do_sample=False, num_beams=1, repetition_penalty=1.1,
                         no_repeat_ngram_size=3, max_new_tokens=MAX_NEW_TOKENS),
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--ckpt", required=True, help="Stage 1 checkpoint")
    ap.add_argument("--manifest", default="data/manifest/dev-clean.tsv")
    ap.add_argument("--out", required=True, help="輸出的 hyp 檔路徑")
    ap.add_argument("--mode", default="nobias", choices=["nobias", "oracle", "stage1", "file"])
    ap.add_argument("--entities", default=None, help="--mode file 用：{utt_id: [entity, ...]} 的 JSON")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 句（快速測試）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decode", default="greedy", choices=list(DECODE),
                    help="greedy（預設）、official（官方 code 的 beam 20 + sampling）或 greedy_norep（greedy + 防重複）")
    ap.add_argument("--amp", default="auto", choices=AMP_CHOICES,
                    help="encoder 精度；auto = 3090/A5000 用 bf16、2080 Ti 用 fp16")
    return ap.parse_args()


@torch.no_grad()
def transcribe_one(model, input_features, entity_list, dev, decode="greedy"):
    """單句生成，回傳文字（小寫、單行）。"""
    feats = input_features.unsqueeze(0).to(dev)         # (1, 80, 3000)
    h, ctc_logits = model.encode_audio(feats)
    audio_tok = ctc_compress(h[0], ctc_logits[0])

    prompt = build_gen_prompt(audio_tok, entity_list,
                              model.tokenizer, model.lm.get_input_embeddings())
    out_ids = model.lm.generate(
        inputs_embeds=prompt["inputs_embeds"],
        attention_mask=prompt["attention_mask"],
        **DECODE[decode],
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    text = model.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Bug A 修正：換行 / tab / 多重空白全部收成單一空格，否則 TSV 一句會變多行
    return " ".join(text.lower().split())


def main():
    args = parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)                  # --decode official 有 sampling，固定亂數才可重現

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    # Bug B 修正：評估不能過濾 >30 秒的句子，測試集少一句 WER 就和論文不可比
    ds = LibriSpeechDataset(args.manifest, fe, max_duration=1e9)

    model = load_stage1_model(args.ckpt, args.stack, dev, args.amp)

    rare_pool = load_rare_pool()
    rng = random.Random(args.seed)
    ent_file = None
    if args.mode == "file":
        assert args.entities, "--mode file 需要 --entities"
        ent_file = json.loads(Path(args.entities).read_text(encoding="utf-8"))
        print(f"從 {args.entities} 讀入 {len(ent_file)} 句的 entity 清單")

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ent = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=args.mode, mininterval=30):
            s = ds[i]
            if args.mode == "nobias":
                el = []
            elif args.mode == "oracle":
                el = s["rare_words"]
            elif args.mode == "stage1":
                el = make_stage1_list(s["rare_words"], rare_pool, rng)
            else:
                el = ent_file.get(s["utt_id"], [])
            n_ent += len(el)

            hyp = transcribe_one(model, s["input_features"], el, dev, args.decode)
            f.write(f"{s['utt_id']}\t{hyp}\n")

            if i < 3:
                print(f"\n  REF: {s['text'][:80]}")
                print(f"  HYP: {hyp[:80]}")
                print(f"  entity {len(el)} 個: {el[:10]}")

    print(f"\n寫入 {out_path}（{n} 句，解碼 {args.decode}，平均每句 {n_ent / max(n, 1):.1f} 個 entity）")
    print(f"峰值 GPU 記憶體 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
