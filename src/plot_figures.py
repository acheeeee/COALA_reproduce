"""從 results/ 畫報告用的圖（在本機或伺服器都可以跑，只需要 matplotlib）。

score_training_<tag>.png    Stage 2 訓練過程：正確答案、干擾字、<unbiased> 的平均分數（每種 loss 一格）
fig3_bwer_<tag>.png         論文 Fig. 3：B-WER vs 每句 target 數（需要 bwer_by_targets.py 的輸出）
fig4_boxplot_<tag>.png      論文 Fig. 4：正負樣本分數的盒狀圖，依每句 target 數分組（需要 fig4_data.py 的輸出）

缺資料的圖會跳過並印出原因。

用法
    python src/plot_figures.py --results results --out figures --tag 100h
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOSSES = [("bias", "Bias-Loss"), ("mpd", "MPD-Loss"), ("dpd", "DPD-Loss"), ("disc", "disc only (extra)")]
POS, NEG, UNB = "#59b84a", "#e08b3a", "#7a5cc7"             # 綠 / 橘（同論文 Fig. 4）/ 紫
STEPS_PER_EPOCH = {"100h": 28539, "960h": 70000}
BUCKETS3 = [("1", "Single\n(1)"), ("2-5", "Medium\n(2-5)"), ("6-24", "Many\n(6-24)")]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": "#dddddd",
                     "axes.spines.top": False, "axes.spines.right": False})


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def smooth(y, k):
    if len(y) < k:
        return np.asarray(y)
    return np.convolve(y, np.ones(k) / k, mode="valid")


def run_dir(tag, loss):
    return f"stage2_{loss}" if tag == "100h" else f"960h__stage2_{loss}"


def score_training(results, out, tag):
    curves = Path(results) / "curves"
    panels = [(l, name) for l, name in LOSSES if (curves / f"{run_dir(tag, l)}.scalars.tsv").exists()]
    if not panels:
        return print(f"   跳過 score_training_{tag}：沒有 {tag} 的 scalars.tsv")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (loss, name) in zip(axes, panels):
        series = defaultdict(list)
        for r in read_tsv(curves / f"{run_dir(tag, loss)}.scalars.tsv"):
            series[r["tag"]].append((int(r["step"]), float(r["value"])))
        for key, label, color in (("train/pos_score_mean", "Positive", POS),
                                  ("train/neg_score_mean", "Negative", NEG),
                                  ("train/unbiased_score", "<unbiased>", UNB)):
            pts = sorted(series[key])
            x = np.array([s for s, _ in pts]) / STEPS_PER_EPOCH[tag]
            y = np.array([v for _, v in pts])
            k = max(1, len(y) // 100)                         # 約 1% 的移動平均，去掉單句的雜訊
            ys = smooth(y, k)
            ax.plot(x[len(x) - len(ys):], ys, color=color, lw=1.8, label=label)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Mean score")
    axes[0].legend(frameon=True, loc="best")
    fig.suptitle(f"Biasing scores during Stage 2 training ({tag})", y=1.02)
    fig.tight_layout()
    path = Path(out) / f"score_training_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   寫入 {path}")


def fig3_bwer(results, out, tag):
    src = Path(results) / "figures_data" / "fig3_bwer_by_targets.tsv"
    if not src.exists():
        return print(f"   跳過 fig3_bwer_{tag}：沒有 {src}（在伺服器跑 bwer_by_targets.py）")
    rows = [r for r in read_tsv(src) if r["tag"] == tag and r["split"] == "all"]
    if not rows:
        return print(f"   跳過 fig3_bwer_{tag}：{src} 裡沒有 {tag}")
    buckets = [str(i) for i in range(1, 11)] + ["11+"]
    styles = {"Oracle": ("#c58be0", "o"), "BTI N=500": ("#8a4fe3", "o"), "BTI N=1000": ("#5b7cf0", "s"),
              "BTI N=5000": ("#2c4aa8", "^"), "w/o Biasing": ("#9e9e9e", "D")}
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for cond, (color, marker) in styles.items():
        val = {r["bucket"]: float(r["b_wer"]) for r in rows if r["condition"] == cond}
        if val:
            ax.plot(buckets, [val.get(b, np.nan) for b in buckets], color=color, marker=marker,
                    ms=4.5, lw=1.8, label=cond)
    ax.set_xlabel("Number of Target Entities")
    ax.set_ylabel("B-WER")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=9)
    fig.tight_layout()
    path = Path(out) / f"fig3_bwer_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   寫入 {path}")


def fig4_boxplot(results, out, tag):
    src = Path(results) / "figures_data" / "fig4_scores.tsv"
    if not src.exists():
        return print(f"   跳過 fig4_boxplot_{tag}：沒有 {src}（在伺服器跑 fig4_data.py）")
    data = defaultdict(list)
    for r in read_tsv(src):
        if r["tag"] == tag:
            data[(r["loss"], r["bucket"], r["label"])].append(float(r["score"]))
    panels = [(l, n) for l, n in LOSSES if any(k[0] == l for k in data)]
    if not panels:
        return print(f"   跳過 fig4_boxplot_{tag}：{src} 裡沒有 {tag}")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (loss, name)) in enumerate(zip(axes, panels)):
        for j, (b, _) in enumerate(BUCKETS3):
            for off, label, color in ((-0.18, "pos", POS), (0.18, "neg", NEG)):
                v = data.get((loss, b, label), [])
                if not v:
                    continue
                bp = ax.boxplot(v, positions=[j + off], widths=0.3, showfliers=False, patch_artist=True,
                                medianprops=dict(color="#555555"), whiskerprops=dict(color="#666666"),
                                capprops=dict(color="#666666"), boxprops=dict(edgecolor="#666666"))
                bp["boxes"][0].set_facecolor(color)
        ax.set_xticks(range(len(BUCKETS3)), [t for _, t in BUCKETS3])
        ax.set_xlim(-0.6, len(BUCKETS3) - 0.4)
        ax.set_xlabel(f"({chr(97 + i)}) {name}", fontsize=12, labelpad=8)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=POS, ec="#666666"), plt.Rectangle((0, 0), 1, 1, fc=NEG, ec="#666666")]
    axes[0].legend(handles, ["Positive", "Negative"], title="Label", loc="best")
    fig.tight_layout()
    path = Path(out) / f"fig4_boxplot_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"   寫入 {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="100h", choices=["100h", "960h"])
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    score_training(args.results, args.out, args.tag)
    fig3_bwer(args.results, args.out, args.tag)
    fig4_boxplot(args.results, args.out, args.tag)


if __name__ == "__main__":
    main()
