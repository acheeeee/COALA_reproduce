"""整理結果紀錄：exp/recall_all.tsv、exp/wer_all.tsv 只留最新的結果。

recall_all.tsv（Table 1）
    同一個打分檔算過好幾次（例如打分還沒跑完就先算了一次）→ 只留最後一次
    檔名不是 *.e6.*（舊的 best.pt 打分、try.jsonl）→ 刪掉
wer_all.tsv（Table 2）
    名稱結尾是 .greedy 的 → 搬到 exp/archive/wer_greedy.tsv（主表改用 greedy_norep，greedy 留作附錄）
    同一個名稱 + 評分方式出現好幾次 → 只留最後一次

整理前會把原檔備份到 exp/archive/。可以重複執行：greedy 的解碼還在跑時，跑完再執行一次即可。

用法（在 repo 根目錄）
    python src/clean_records.py --dry-run     # 只印出會怎麼改
    python src/clean_records.py
"""

import argparse
import shutil
import time
from pathlib import Path

RECALL = Path("exp/recall_all.tsv")
WER = Path("exp/wer_all.tsv")
ARCHIVE = Path("exp/archive")
WER_GREEDY = ARCHIVE / "wer_greedy.tsv"


def read_tsv(path):
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    header = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:] if l.split("\t") != header]
    return header, rows


def write_tsv(path, header, rows):
    path.write_text("\n".join("\t".join(r) for r in [header] + rows) + "\n", encoding="utf-8")


def latest(rows, key):
    """同一個 key 只留最後出現的那一列，順序照最後出現的位置。"""
    last = {key(r): i for i, r in enumerate(rows)}
    return [r for i, r in enumerate(rows) if last[key(r)] == i]


def clean_recall(dry):
    if not RECALL.exists():
        print(f"{RECALL} 不存在，跳過")
        return
    header, rows = read_tsv(RECALL)
    keep = latest([r for r in rows if ".e6." in r[0]], key=lambda r: r[0])
    print(f"\n== {RECALL}：{len(rows)} 列 → {len(keep)} 列")
    for r in rows:
        if r not in keep:
            print(f"   刪掉  {r[0]}（{r[1]} 句）")
    for r in keep:
        print(f"   保留  {r[0]}（{r[1]} 句）")
    if not dry:
        write_tsv(RECALL, header, sorted(keep, key=lambda r: r[0]))


def clean_wer(dry):
    if not WER.exists():
        print(f"{WER} 不存在，跳過")
        return
    header, rows = read_tsv(WER)
    key = lambda r: (r[0], r[1])
    greedy = [r for r in rows if r[0].endswith(".greedy")]
    keep = latest([r for r in rows if not r[0].endswith(".greedy")], key=key)
    print(f"\n== {WER}：{len(rows)} 列 → {len(keep)} 列（{len(greedy)} 列 greedy 搬到 {WER_GREEDY}）")
    for r in keep:
        print(f"   保留  {r[0]:<45} {r[1]:<9} B-WER {r[2]}")
    if not dry:
        old = read_tsv(WER_GREEDY)[1] if WER_GREEDY.exists() else []
        write_tsv(WER_GREEDY, header, latest(old + greedy, key=key))
        write_tsv(WER, header, keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只印出會怎麼改，不寫檔")
    args = ap.parse_args()

    if not args.dry_run:
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M")
        for p in (RECALL, WER):
            if p.exists():
                shutil.copy(p, ARCHIVE / f"{p.stem}.{stamp}.bak.tsv")
        print(f"原檔備份到 {ARCHIVE}/*.{stamp}.bak.tsv")

    clean_recall(args.dry_run)
    clean_wer(args.dry_run)
    if args.dry_run:
        print("\n（--dry-run：沒有寫任何檔案）")


if __name__ == "__main__":
    main()
