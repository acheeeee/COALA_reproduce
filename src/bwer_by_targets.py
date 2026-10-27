"""論文 Fig. 3 的資料：依「每句 target 數」分組算 B-WER（fbai 定義）。只用 CPU，幾十秒。

每句的 target 數 = fbai ref 檔第 3 欄的字數；分組 1、2、…、10、11+。
條件：w/o Biasing、Oracle、BTI N=500 / 1000 / 5000（防重複解碼的逐字稿，缺的條件會跳過）。
split = test-clean、test-other、all（兩者合併，論文 Fig. 3 的畫法）。

用法（在 repo 根目錄）
    python src/bwer_by_targets.py --tag 100h
    python src/bwer_by_targets.py --tag 960h       # 附加到同一個檔
輸出 exp/fig3_bwer_by_targets.tsv（同一個 tag 會先刪掉舊的再寫）
"""

import argparse
from collections import defaultdict
from pathlib import Path

from score_wer import FBAI_SCORE, count, load_fbai, rate, read_hyps, read_refs

REF = "/mnt/disk2/M11515125/fbai-speech/is21_deep_bias/ref/{split}.biasing_1000.tsv"
CONDITIONS = {
    "w/o Biasing": "{tag}.{split}.nobias.greedy_norep.tsv",
    "Oracle": "{tag}.{split}.oracle.greedy_norep.tsv",
    "BTI N=500": "{tag}.{split}.bti_dpd.N500.greedy_norep.tsv",
    "BTI N=1000": "{tag}.{split}.bti_dpd.N1000.greedy_norep.tsv",
    "BTI N=5000": "{tag}.{split}.bti_dpd.N5000.greedy_norep.tsv",
}
OUT = Path("exp/fig3_bwer_by_targets.tsv")
HEADER = ["tag", "condition", "split", "bucket", "utts", "biasing_words", "b_wer"]


def bucket(n):
    return str(n) if n <= 10 else "11+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="100h")
    ap.add_argument("--hyp-dir", default="exp/hyp")
    args = ap.parse_args()

    fb = load_fbai(FBAI_SCORE)
    rows = []
    for cond, pattern in CONDITIONS.items():
        groups = defaultdict(list)                        # (split, bucket) → items
        for split in ("test-clean", "test-other"):
            hyp_path = Path(args.hyp_dir) / pattern.format(tag=args.tag, split=split)
            if not hyp_path.exists():
                print(f"   跳過 {cond} {split}：沒有 {hyp_path}")
                continue
            refs, hyps = read_refs(REF.format(split=split)), read_hyps(hyp_path)
            for u, (text, bw) in refs.items():
                if u not in hyps or not bw:
                    continue
                item = (text.lower().split(), hyps[u].lower().split(), {w.lower() for w in bw})
                groups[(split, bucket(len(bw)))].append(item)
                groups[("all", bucket(len(bw)))].append(item)
        for (split, b), items in sorted(groups.items()):
            _, _, bwer = count(fb, items)
            rows.append([args.tag, cond, split, b, len(items), bwer.ref_words, f"{rate(bwer):.2f}"])
        if groups:
            done = sorted((r for r in rows if r[1] == cond and r[2] == "all"),
                          key=lambda r: 99 if r[3] == "11+" else int(r[3]))
            print(f"{cond:<12} " + "  ".join(f"{r[3]}:{r[6]}" for r in done))

    old = []
    if OUT.exists():
        old = [l.rstrip("\n").split("\t") for l in open(OUT, encoding="utf-8")][1:]
        old = [r for r in old if r[0] != args.tag]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in old + rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(f"\n寫入 {OUT}（{args.tag} {len(rows)} 列）")


if __name__ == "__main__":
    main()
