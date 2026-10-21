"""C-11：Biasing Target Identification（論文 3.4 推論段落）。

每句取分數最高的 K=10 個候選，再用 <unbiased> 的分數當門檻，低於它的丟掉
（論文：99% 的句子 rare word 少於 10 個，所以 top-10 之外還要再過濾負樣本）。
輸出 {utt_id: [entity, ...]} 的 JSON，交給 evaluate.py --mode file 生成轉寫。

--official：官方 code（recipes/inference.py 第 239 行）的做法 ——
    把 <unbiased> 和所有候選一起排序取前 10，**不過濾**；<unbiased> 若擠進前 10 就佔掉一格。
    官方會把 <unbiased> 也寫進 ASR prompt，但我們的 Stage 1 沒看過它，所以這裡拿掉它，只留真的詞
    → 結果是「前 10 名（<unbiased> 不在前 10）」或「前 9 名（<unbiased> 在前 10）」。

用法
    python src/bti.py exp/score/dpd.test-clean.N5000.jsonl --out exp/bti/dpd.test-clean.N5000.json
    python src/evaluate.py --ckpt exp/stage1_tc100_s2/best.pt --manifest data/manifest/test-clean.tsv \\
        --mode file --entities exp/bti/dpd.test-clean.N5000.json --out exp/hyp/test-clean.bti_dpd.N5000.tsv
"""

import argparse
import json
from pathlib import Path


def select_official(row, top_k):
    """官方做法：<unbiased> 一起排序取前 top_k，不過濾；<unbiased> 在前 top_k 名就少取一個真的詞。"""
    k = top_k if row["n_above_unb"] >= top_k else top_k - 1
    return [entity for entity, _, _ in row["top"][:k]]


def select(row, top_k, use_threshold):
    """一句的 BTI 結果。row["top"] 已經依分數由高到低排好：[[entity, score, is_pos], ...]"""
    picked = []
    for entity, score, _ in row["top"][:top_k]:
        if use_threshold and score <= row["s_unbiased"]:
            break                                  # 已排序，後面只會更低
        picked.append(entity)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", help="score_stage2.py 輸出的 JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--no-threshold", action="store_true", help="只取 top-K，不用 <unbiased> 門檻（消融用）")
    ap.add_argument("--official", action="store_true", help="官方 code 的做法（見檔案開頭說明）")
    args = ap.parse_args()

    with open(args.scores, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert all(len(r["top"]) >= args.top_k or len(r["top"]) == r["n_cand"] for r in rows), \
        "JSONL 存的 top 數量比 --top-k 少，請用更大的 --save-top 重跑 score_stage2.py"

    out, n_sel, hit, n_pos, n_empty = {}, 0, 0, 0, 0
    for r in rows:
        picked = (select_official(r, args.top_k) if args.official
                  else select(r, args.top_k, not args.no_threshold))
        out[r["utt_id"]] = picked
        pos = set(r["positives"])
        n_sel += len(picked)
        hit += len(pos & set(picked))
        n_pos += len(pos)
        n_empty += not picked

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    mode = (f"官方做法：<unbiased> 一起排序取 top-{args.top_k}，不過濾" if args.official
            else f"top-{args.top_k}" + ("" if args.no_threshold else " + <unbiased> 門檻"))
    print(f"{args.scores} → {args.out}（{mode}）")
    print(f"   {len(rows)} 句，平均每句選 {n_sel / len(rows):.2f} 個，{n_empty} 句（{100 * n_empty / len(rows):.1f}%）一個都沒選")
    print(f"   選中的正樣本 {hit}/{n_pos}（recall {100 * hit / max(n_pos, 1):.2f}%），"
          f"precision {100 * hit / max(n_sel, 1):.2f}%")


if __name__ == "__main__":
    main()
