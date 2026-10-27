"""論文 Fig. 4 的資料：正負樣本的分數分布，依每句 target 數分成 Single (1) / Medium (2-5) / Many (6-24)。

一般的打分檔只存前 50 名，負樣本只剩最難的那幾個，畫不出完整的分布，所以另外抽一小批句子，
存下全部 5000 個候選的分數。

步驟（在 repo 根目錄）
    1. python src/fig4_data.py subset
       每個 split、每組各隨機抽 --per-bucket 句（預設 80）→ data/manifest/fig4.test-clean.tsv、fig4.test-other.tsv
    2. 打分（每種 loss × 每個 split），--save-top 99999 = 存下全部候選：
       python src/score_stage2.py --ckpt exp/stage2_dpd/epoch6.pt --split test-clean \\
           --manifest data/manifest/fig4.test-clean.tsv --n 5000 --save-top 99999 --out exp/fig4/100h/dpd.test-clean.jsonl
    3. python src/fig4_data.py collect --tag 100h
       正樣本全部留下、負樣本每句隨機留 --neg-per-utt 個（預設 50）→ exp/fig4_scores.tsv（幾 MB，可上傳）
"""

import argparse
import json
import random
from pathlib import Path

LOSSES = ("bias", "mpd", "dpd", "disc")
SPLITS = ("test-clean", "test-other")
OUT = Path("exp/fig4_scores.tsv")
HEADER = ["tag", "loss", "split", "bucket", "label", "score"]


def bucket(n):
    if n == 1:
        return "1"
    if 2 <= n <= 5:
        return "2-5"
    if 6 <= n <= 24:
        return "6-24"
    return None                                            # 0 個或 >24 個：不畫


def subset(args):
    rng = random.Random(args.seed)
    for split in SPLITS:
        src = Path(f"data/manifest/{split}.tsv")
        lines = src.read_text(encoding="utf-8").splitlines()
        header = lines[0].split("\t")
        col = header.index("rare_words")
        groups = {}
        for l in lines[1:]:
            b = bucket(len(json.loads(l.split("\t")[col])))
            if b:
                groups.setdefault(b, []).append(l)
        keep = []
        for b in ("1", "2-5", "6-24"):
            pick = rng.sample(groups[b], min(args.per_bucket, len(groups[b])))
            keep += pick
            print(f"{split} {b:>4}：{len(groups[b])} 句 → 抽 {len(pick)} 句")
        dst = Path(f"data/manifest/fig4.{split}.tsv")
        dst.write_text("\n".join([lines[0]] + keep) + "\n", encoding="utf-8")
        print(f"   寫入 {dst}")


def collect(args):
    rng = random.Random(args.seed)
    rows = []
    for loss in LOSSES:
        for split in SPLITS:
            path = Path(f"exp/fig4/{args.tag}/{loss}.{split}.jsonl")
            if not path.exists():
                print(f"   跳過 {path}（還沒打分）")
                continue
            n_utt = n_pos = 0
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                b = bucket(len(r["positives"]))
                if not b:
                    continue
                if len(r["top"]) < r["n_cand"]:
                    raise SystemExit(f"{path} 只存了前 {len(r['top'])} 名，打分時要加 --save-top 99999")
                pos = [s for _, s, p in r["top"] if p]
                neg = [s for _, s, p in r["top"] if not p]
                neg = rng.sample(neg, min(args.neg_per_utt, len(neg)))
                rows += [[args.tag, loss, split, b, "pos", f"{s:.4f}"] for s in pos]
                rows += [[args.tag, loss, split, b, "neg", f"{s:.4f}"] for s in neg]
                n_utt += 1
                n_pos += len(pos)
            print(f"{loss:<5} {split}：{n_utt} 句，正樣本 {n_pos} 個")

    old = []
    if OUT.exists():
        old = [l.rstrip("\n").split("\t") for l in open(OUT, encoding="utf-8")][1:]
        old = [r for r in old if r[0] != args.tag]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in old + rows:
            f.write("\t".join(r) + "\n")
    print(f"\n寫入 {OUT}（{args.tag} {len(rows):,} 列）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["subset", "collect"])
    ap.add_argument("--per-bucket", type=int, default=80)
    ap.add_argument("--neg-per-utt", type=int, default=50)
    ap.add_argument("--tag", default="100h")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    subset(args) if args.step == "subset" else collect(args)


if __name__ == "__main__":
    main()
