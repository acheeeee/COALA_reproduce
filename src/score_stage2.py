"""C-9：用 Stage 2 模型為測試集每句的 biasing list 打分，輸出給 recall.py / bti.py。

清單來源
    N = 100 / 500 / 1000 / 2000：官方 ref/{split}.biasing_{N}.tsv
    其他 N（論文的 5000）：自己抽，每句用 Random(f"{seed}-{utt_id}") → 和句子順序無關、可重現

每句的候選 = [<unbiased>] + biasing list。N=5000 時一次 5000 條尾巴會 OOM，所以：
    音訊 + 前綴只算一次 → 前綴的 KV cache 複製給每一段（--chunk 條）→ 分段算完再接起來
數學上和 model.forward_stage2 完全相同（--check 會驗證）。

輸出 JSONL，每句一行：
    utt_id, n_cand（不含 <unbiased>）, positives,
    pos_ranks   每個正樣本在「不含 <unbiased> 的候選」裡的名次（1 = 最高分）
    s_unbiased  <unbiased> 的分數（BTI 的門檻）
    n_above_unb 分數高於 <unbiased> 的候選數
    top         前 --save-top 名 [[entity, score], ...]

用法
    python src/score_stage2.py --ckpt exp/stage2_dpd/best.pt --split test-clean --n 500 \\
        --out exp/score/dpd.test-clean.N500.jsonl
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import WhisperFeatureExtractor

from biasing import add_unbiased, load_official_ref, load_rare_pool, make_biasing_list
from dataset import LibriSpeechDataset
from eval_utils import AMP_CHOICES, load_stage2_model
from model import WHISPER_ID
from modules import ctc_compress
from prompt import build_stage2_sample

OFFICIAL_N = {100, 500, 1000, 2000}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Stage 2 checkpoint（例 exp/stage2_dpd/best.pt）")
    ap.add_argument("--stage1-ckpt", default=None, help="預設用 Stage 2 checkpoint 裡記錄的路徑")
    ap.add_argument("--split", default="test-clean")
    ap.add_argument("--manifest", default=None, help="預設 data/manifest/{split}.tsv")
    ap.add_argument("--n", type=int, required=True, help="biasing list 大小 N")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=128, help="一次送幾條尾巴；某句 OOM 時那句自動減半")
    ap.add_argument("--save-top", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="自己抽清單（N 不是官方大小）時用")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--amp", default="auto", choices=AMP_CHOICES)
    ap.add_argument("--dump-lists", default=None,
                    help="另外把每句完整的 biasing list 存成 JSON（給 evaluate.py --mode file 重現 OOM）")
    ap.add_argument("--check", action="store_true",
                    help="先用第一句驗證：分段版 == model.forward_stage2（K=121）")
    return ap.parse_args()


def _fresh_cache(model, prefix, base_cache):
    """前綴 KV cache 的新複本。deepcopy 失敗（不同 transformers 版本的 cache 物件）就重算一次前綴。"""
    try:
        return copy.deepcopy(base_cache)
    except Exception:
        return model.lm(inputs_embeds=prefix.unsqueeze(0), use_cache=True).past_key_values


@torch.no_grad()
def score_candidates(model, input_features, candidates, chunk):
    """一句話 + 任意多個候選 → (K,) 分數。音訊與前綴只算一次，候選分段送。"""
    h, ctc_logits = model.encode_audio(input_features)
    audio_tok = ctc_compress(h[0], ctc_logits[0])
    embed = model.lm.get_input_embeddings()

    # 前綴對所有候選都一樣：先單獨算一次，拿到 KV cache
    prefix = build_stage2_sample(audio_tok, candidates[:1], model.tokenizer, embed)["prefix_embeds"]
    P = prefix.shape[0]
    base_cache = model.lm(inputs_embeds=prefix.unsqueeze(0), use_cache=True).past_key_values

    # 長句的前綴長，複製 chunk 份 KV cache 可能 OOM → 只對這一句把 chunk 減半重試
    # （chunk 小反而慢：forward 次數變多，2080 Ti 上 32 比 128 慢約 3 倍，所以不全域調小）
    scores, i = [], 0
    while i < len(candidates):
        try:
            s = _score_chunk(model, audio_tok, embed, prefix, base_cache, candidates[i:i + chunk])
        except torch.cuda.OutOfMemoryError:
            if chunk == 1:
                raise
            s = None
        if s is None:                                     # 離開 except 後才釋放，失敗那段的張量才真的放掉
            torch.cuda.empty_cache()
            chunk //= 2
            tqdm.write(f"  OOM（前綴 {P} token）→ chunk 改為 {chunk}")
            continue
        scores.append(s)
        i += s.shape[0]
    return torch.cat(scores)


def _score_chunk(model, audio_tok, embed, prefix, base_cache, part):
    P, k = prefix.shape[0], len(part)
    s2 = build_stage2_sample(audio_tok, part, model.tokenizer, embed)
    cache = _fresh_cache(model, prefix, base_cache)   # forward 會就地加長 cache，每段要用新的複本
    cache.batch_repeat_interleave(k)
    mask = torch.cat([torch.ones(k, P, dtype=s2["suffix_mask"].dtype, device=prefix.device),
                      s2["suffix_mask"]], dim=1)
    out = model.lm(inputs_embeds=s2["suffix_embeds"], attention_mask=mask,
                   past_key_values=cache, output_hidden_states=True)
    return model.projector(out.hidden_states[-1], s2["suffix_mask"].float())


def build_lists(split, n, ds, seed):
    """回傳 {utt_id: (positives, biasing_list)}。官方大小讀 ref，其他自己抽。"""
    if n in OFFICIAL_N:
        ref = load_official_ref(split, n)
        extra = [len(r["biasing_list"]) - len(set(r["positives"]) & set(r["biasing_list"]))
                 for r in ref.values()]
        print(f"官方 {split}.biasing_{n}.tsv：{len(ref)} 句，干擾詞數 平均 {sum(extra) / len(extra):.1f}"
              f"（min {min(extra)} / max {max(extra)}）")
        return {u: (r["positives"], r["biasing_list"]) for u, r in ref.items()}

    pool = load_rare_pool()
    out = {}
    for row in ds.rows:
        pos = row["rare_words"]
        rng = random.Random(f"{seed}-{row['utt_id']}")
        out[row["utt_id"]] = (pos, make_biasing_list(pos, pool, n, rng))
    print(f"自己抽 N={n} 的清單（seed={seed}，每句正樣本 + {n} 個干擾詞）：{len(out)} 句")
    return out


def summarize(utt_id, candidates, positives, s, save_top):
    """把一句的分數整理成要寫進 JSONL 的 dict。candidates[0] 是 <unbiased>。"""
    s_unb = s[0].item()
    real_s = s[1:]
    real = candidates[1:]
    order = torch.argsort(real_s, descending=True).tolist()
    rank_of = {real[j]: r + 1 for r, j in enumerate(order)}
    pos_set = set(positives)
    return {
        "utt_id": utt_id,
        "n_cand": len(real),
        "positives": positives,
        "pos_ranks": [rank_of[p] for p in positives if p in rank_of],
        "s_unbiased": round(s_unb, 5),
        "n_above_unb": int((real_s > s_unb).sum()),
        "top": [[real[j], round(real_s[j].item(), 5), real[j] in pos_set] for j in order[:save_top]],
    }


def main():
    args = parse_args()
    dev = torch.device("cuda")
    manifest = args.manifest or f"data/manifest/{args.split}.tsv"

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset(manifest, fe, max_duration=1e9)
    model = load_stage2_model(args.ckpt, dev, args.stage1_ckpt, args.amp)
    lists = build_lists(args.split, args.n, ds, args.seed)

    if args.dump_lists:
        Path(args.dump_lists).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_lists).write_text(
            json.dumps({u: bl for u, (_, bl) in lists.items()}, ensure_ascii=False), encoding="utf-8")
        print(f"完整清單 → {args.dump_lists}")

    if args.check:
        smp = ds[0]
        pos, bl = lists[smp["utt_id"]]
        cands = add_unbiased(bl)[:121]
        x = smp["input_features"].unsqueeze(0).to(dev)
        with torch.no_grad():
            ref = model.forward_stage2(x, cands)["s"]
        mine = score_candidates(model, x, cands, chunk=50)       # 故意切成 3 段
        d = (ref - mine).abs().max().item()
        print(f"[check] 分段版 vs forward_stage2（K={len(cands)}，chunk=50）最大差 {d:.2e}"
              f"  {'✅' if d < 1e-4 else '❌'}")
        assert d < 1e-4, "分段打分和 forward_stage2 不一致"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():                                     # 中斷後重跑：跳過已完成的句子
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["utt_id"] for line in f if line.strip()}
        print(f"{out_path} 已有 {len(done)} 句，接著跑")

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    t0, n_run, missing = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(n), desc=f"{args.split} N={args.n}", mininterval=30):
            utt_id = ds.rows[i]["utt_id"]
            if utt_id in done:
                continue
            if utt_id not in lists:
                missing += 1
                continue
            smp = ds[i]
            positives, bl = lists[utt_id]
            cands = add_unbiased(bl)
            s = score_candidates(model, smp["input_features"].unsqueeze(0).to(dev), cands, args.chunk)
            f.write(json.dumps(summarize(utt_id, cands, positives, s, args.save_top),
                               ensure_ascii=False) + "\n")
            n_run += 1

    sec = (time.time() - t0) / max(n_run, 1)
    print(f"\n完成 {n_run} 句 → {out_path}（{sec:.2f} 秒/句，峰值 GPU "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB）")
    if missing:
        print(f"⚠️ {missing} 句不在清單裡（manifest 和 ref 的句子不一致）")


if __name__ == "__main__":
    main()
