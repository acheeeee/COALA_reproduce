"""把 exp/ 裡要上傳的結果整理到 results/（exp/ 在 .gitignore 裡，results/ 不在）。

results/
    recall_all.tsv、wer_all.tsv        正式紀錄（先跑 clean_records.py）
    wer_greedy.tsv                     greedy 的 WER（附錄）
    score/   exp/score/*.e6.*.jsonl    Stage 2 打分
    hyp/     exp/hyp/*.tsv             解碼出來的逐字稿
    bti/     exp/bti/*.json            BTI 挑出的字
    curves/  <run>.epoch_summary.tsv   每個 epoch 的摘要
             <run>.scalars.tsv         tensorboard 的逐步紀錄（tag / step / value），每條曲線最多 5000 點，取代 50 MB 的 events 檔
    logs/    <log>.log                 進度條只留每 200 行一行和每段的最後一行，其他原樣
    figures_data/                      論文 Fig. 3、Fig. 4 的資料（bwer_by_targets.py、fig4_data.py 的輸出）
    MANIFEST.tsv                       每個檔的來源、行數、來源最後修改時間 → 用來判斷是不是最新的

不複製：模型檔（*.pt）、exp/lists/（可重新產生，N5000 超過 GitHub 單檔 100 MB）、exp/archive/*.bak.tsv、
冒煙測試（smoke）、try*。每次執行都會整個重建 results/。

用法（在 repo 根目錄）
    python src/clean_records.py && python src/collect_results.py
    git add results && git commit -m "results: ..." && git push origin c_layer_eval
"""

import re
import shutil
import time
from pathlib import Path

EXP = Path("exp")
OUT = Path("results")
BAR = re.compile(r"\d+%\|.*\|\s*\d+/\d+ \[")         # tqdm 進度條
KEEP_EVERY = 200
MAX_POINTS = 5000                                 # 每條曲線最多留幾個點（畫圖夠用）
SKIP = ("archive", "smoke", "lists", "_try")

manifest = []


def skipped(p):
    return any(k in part for part in p.parts for k in SKIP) or p.name.startswith("try")


def run_name(d):
    """exp/960h/stage2_dpd → 960h__stage2_dpd"""
    return "__".join(d.relative_to(EXP).parts) or "exp"


def record(dst, src):
    n = sum(1 for _ in open(dst, encoding="utf-8", errors="replace"))
    mtime = time.strftime("%m-%d_%H:%M", time.localtime(src.stat().st_mtime))
    manifest.append((str(dst.relative_to(OUT)), str(src), n, mtime))


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    record(dst, src)


def squeeze_log(src, dst):
    text = src.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = [l for l in text.split("\n") if l.strip()]
    out, run = [], []
    for l in lines + [""]:                            # 最後補一行空的，讓最後一段進度條也結算
        if BAR.search(l):
            run.append(l)
            continue
        if run:                                       # 一段連續的進度條：每 KEEP_EVERY 行留一行 + 最後一行
            out += run[::KEEP_EVERY] + ([run[-1]] if (len(run) - 1) % KEEP_EVERY else [])
            run = []
        if l:
            out.append(l)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    record(dst, src)


def export_scalars(d, dst):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print(f"   （沒有 tensorboard，跳過 {d}）")
        return
    ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
    ea.Reload()
    with open(dst, "w", encoding="utf-8") as f:
        f.write("tag\tstep\tvalue\n")
        for tag in sorted(ea.Tags()["scalars"]):
            ev = ea.Scalars(tag)
            stride = max(1, -(-len(ev) // MAX_POINTS))      # 太多點就等距抽樣，最後一點一定留
            for e in ev[::stride] + ([ev[-1]] if (len(ev) - 1) % stride else []):
                f.write(f"{tag}\t{e.step}\t{e.value:.6g}\n")
    record(dst, next(d.glob("events.out.tfevents.*")))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    for name in ("recall_all.tsv", "wer_all.tsv"):
        if (EXP / name).exists():
            copy(EXP / name, OUT / name)
    if (EXP / "archive/wer_greedy.tsv").exists():
        copy(EXP / "archive/wer_greedy.tsv", OUT / "wer_greedy.tsv")

    for name in ("fig3_bwer_by_targets.tsv", "fig4_scores.tsv"):          # 畫論文 Fig. 3、Fig. 4 用
        if (EXP / name).exists():
            copy(EXP / name, OUT / "figures_data" / name)

    for pattern, sub in (("score/*.e6.*.jsonl", "score"), ("hyp/*.tsv", "hyp"), ("bti/*.json", "bti")):
        for p in sorted(EXP.glob(pattern)):
            if not skipped(p):
                copy(p, OUT / sub / p.name)

    for p in sorted(EXP.rglob("epoch_summary.tsv")):
        if not skipped(p):
            copy(p, OUT / "curves" / f"{run_name(p.parent)}.epoch_summary.tsv")
    for d in sorted({p.parent for p in EXP.rglob("events.out.tfevents.*")}):
        if not skipped(d):
            print(f"   匯出 {d} 的 tensorboard 紀錄")
            export_scalars(d, OUT / "curves" / f"{run_name(d)}.scalars.tsv")

    for p in sorted(EXP.rglob("*.log")):
        if not skipped(p):
            squeeze_log(p, OUT / "logs" / ("__".join(p.relative_to(EXP).parts)))

    with open(OUT / "MANIFEST.tsv", "w", encoding="utf-8") as f:
        f.write("file\tsource\tlines\tsource_mtime\n")
        for row in sorted(manifest):
            f.write("\t".join(map(str, row)) + "\n")

    files = [p for p in OUT.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"\n寫入 {OUT}/：{len(files)} 個檔，共 {total / 2**20:.1f} MB")
    for p in sorted(files, key=lambda p: -p.stat().st_size)[:5]:
        print(f"   {p.stat().st_size / 2**20:6.1f} MB  {p}")
    big = [p for p in files if p.stat().st_size > 50 * 2**20]
    if big:
        print("⚠️ 超過 50 MB（GitHub 會警告，100 MB 會拒絕）：" + "、".join(map(str, big)))


if __name__ == "__main__":
    main()
