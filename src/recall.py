"""C-10：從 score_stage2.py 的 JSONL 算論文 Table 1 的 Recall@X 與 Recall#X。

論文 4.2 的定義：
    Recall@X  每句「至少要取前幾名，才能找回 X% 的正樣本」，再對句子取平均（越小越好）
    Recall#X  每句取前 X 名時，找回的正樣本比例（越大越好）

論文沒交代、這裡自己決定的地方（寫 thesis 時要註明）：
    1. 沒有 rare word 的句子不計入（沒有正樣本，recall 沒有定義）
    2. Recall#X 用「全部句子合計」：Σ 找回數 / Σ 正樣本數（micro）。另外印出逐句平均（macro）當參考
    3. 名次不含 <unbiased>（它不是真的詞）
    4. 某個正樣本根本不在清單裡時，Recall@X 以「整份清單長度」計

⚠️ 官方 code（Guo0911/COALA scripts/evaluation.py 的 recall_score）的 Recall@X 和論文文字不同：
    把「所有句子的所有正樣本名次」放在一起排序，取第 ceil(總數 × X%) 個（0-based 索引）
    = 全體正樣本名次的第 X 百分位，不是逐句平均。論文表格的數字是用這個算的。
    → 這裡兩種都算：Recall@X（論文文字）與 Recall@X·官方（官方 code）

用法
    python src/recall.py exp/score/dpd.test-clean.N5000.jsonl exp/score/dpd.test-other.N5000.jsonl
"""

import argparse
import json
import math
from pathlib import Path

AT_X = [90, 95, 99]
PAPER_TABLE1 = {          # Recall@95 clean/other, Recall#20 clean/other, Recall#50 clean/other
    "Bias-Loss": (10.0, 19.0, 98.72, 95.10, 99.29, 97.27),
    "MPD-Loss": (9.0, 16.9, 98.76, 95.65, 99.43, 97.55),
    "DPD-Loss": (9.0, 13.0, 99.09, 96.59, 99.60, 98.17),
}
HASH_X = [1, 5, 10, 20, 50]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at(rows, x):
    """Recall@X：平均需要前幾名才能找回 X% 的正樣本。"""
    need = []
    for r in rows:
        n = len(r["positives"])
        if n == 0:
            continue
        k = math.ceil(x / 100 * n)              # 至少要找回幾個
        ranks = sorted(r["pos_ranks"])
        need.append(ranks[k - 1] if len(ranks) >= k else r["n_cand"])
    return sum(need) / len(need)


def recall_at_official(rows, x):
    """Recall@X 的官方 code 算法：全體正樣本名次排序後，取索引 ceil(總數 × X%) 的那個（0-based）。"""
    ranks = sorted(k for r in rows for k in r["pos_ranks"])
    idx = min(math.ceil(len(ranks) * x / 100), len(ranks) - 1)
    return ranks[idx]


def recall_hash(rows, x):
    """Recall#X：每句取前 X 名 → (micro %, macro %)。"""
    hit = tot = 0
    per_utt = []
    for r in rows:
        n = len(r["positives"])
        if n == 0:
            continue
        h = sum(1 for k in r["pos_ranks"] if k <= x)
        hit += h
        tot += n
        per_utt.append(h / n)
    return 100 * hit / tot, 100 * sum(per_utt) / len(per_utt)


def unbiased_stats(rows):
    """<unbiased> 門檻的行為：沒有 rare word 的句子，模型有幾成判斷「全部候選都低於 <unbiased>」。"""
    empty = [r for r in rows if not r["positives"]]
    correct = sum(1 for r in empty if r["n_above_unb"] == 0)
    above = sum(r["n_above_unb"] for r in rows) / len(rows)
    return len(empty), (100 * correct / len(empty) if empty else float("nan")), above


def report(path):
    rows = load(path)
    n_pos_utt = sum(1 for r in rows if r["positives"])
    n_cand = sum(r["n_cand"] for r in rows) / len(rows)
    print(f"\n== {path}")
    print(f"   {len(rows)} 句（{n_pos_utt} 句有 rare word），平均 {n_cand:.0f} 個候選")

    at = {x: recall_at(rows, x) for x in AT_X}
    at_off = {x: recall_at_official(rows, x) for x in AT_X}
    hs = {x: recall_hash(rows, x) for x in HASH_X}
    print("   " + "  ".join(f"Recall@{x}={at[x]:.1f}" for x in AT_X) + "   ← 論文文字定義（逐句平均）")
    print("   " + "  ".join(f"Recall@{x}={at_off[x]}" for x in AT_X) + "   ← 官方 code 定義（全體第 X 百分位）")
    print("   " + "  ".join(f"Recall#{x}={hs[x][0]:.2f}%" for x in HASH_X))
    print("   (macro) " + "  ".join(f"#{x}={hs[x][1]:.2f}%" for x in HASH_X))

    n_empty, acc, above = unbiased_stats(rows)
    print(f"   <unbiased>：{n_empty} 句無 rare word，其中 {acc:.1f}% 全部候選都低於門檻；"
          f"平均每句 {above:.1f} 個候選高於門檻")
    return {"file": str(path), "utts": len(rows), "cand": round(n_cand),
            **{f"R@{x}": round(at[x], 2) for x in AT_X},
            **{f"R@{x}_official": at_off[x] for x in AT_X},
            **{f"R#{x}": round(hs[x][0], 2) for x in HASH_X}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="score_stage2.py 輸出的 JSONL")
    ap.add_argument("--tsv", default=None, help="把結果附加到這個 TSV（彙整多組實驗用）")
    args = ap.parse_args()

    results = [report(f) for f in args.files]
    if args.tsv:
        p = Path(args.tsv)
        new = not p.exists()
        with open(p, "a", encoding="utf-8") as f:
            if new:
                f.write("\t".join(results[0].keys()) + "\n")
            for r in results:
                f.write("\t".join(str(v) for v in r.values()) + "\n")
        print(f"\n附加到 {p}")

    print("\n論文 Table 1（N=5000，clean / other）")
    for name, row in PAPER_TABLE1.items():
        print(f"   {name:<10} Recall@95 {row[0]} / {row[1]}   Recall#20 {row[2]} / {row[3]}   "
              f"Recall#50 {row[4]} / {row[5]}")


if __name__ == "__main__":
    main()
