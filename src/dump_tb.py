"""從 tensorboard events 檔還原 epoch 摘要。
用於第一次訓練（stack=4），當時還沒有 epoch_summary.tsv。
"""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

STEPS_PER_EPOCH = 7134  # train-clean-100, batch 4, drop_last


def load_scalars(logdir):
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    print(f"找到 {len(tags)} 個 scalar: {tags}\n")
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir")
    ap.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    args = ap.parse_args()

    sc = load_scalars(args.logdir)

    # 1. 原始資料全部倒成長格式，方便自己畫圖
    raw = Path(args.logdir) / "scalars_raw.tsv"
    with open(raw, "w", encoding="utf-8") as f:
        f.write("tag\tstep\tvalue\n")
        for tag, pts in sorted(sc.items()):
            for step, v in pts:
                f.write(f"{tag}\t{step}\t{v:.6f}\n")
    print(f"原始資料 -> {raw}")

    # 2. 重建 epoch 摘要：train 取該 epoch 內平均、val 取該 epoch 內最後一筆
    def epoch_of(step):
        return (step - 1) // args.steps_per_epoch

    tr_keys = ["train/total_loss", "train/ctc_loss", "train/ce_loss"]
    val_keys = ["val/total_loss", "val/ctc_loss", "val/ce_loss"]

    n_ep = max(epoch_of(s) for pts in sc.values() for s, _ in pts) + 1

    out = Path(args.logdir) / "epoch_summary_recovered.tsv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("epoch\tstep\ttr_total\ttr_ctc\ttr_ce\t"
                "val_total\tval_ctc\tval_ce\tlr\n")
        for ep in range(n_ep):
            row = [str(ep), str((ep + 1) * args.steps_per_epoch)]

            for k in tr_keys:  # 平均
                vals = [v for s, v in sc.get(k, []) if epoch_of(s) == ep]
                row.append(f"{sum(vals) / len(vals):.4f}" if vals else "-")

            for k in val_keys:  # 最後一筆
                vals = [v for s, v in sc.get(k, []) if epoch_of(s) == ep]
                row.append(f"{vals[-1]:.4f}" if vals else "-")

            lrs = [v for s, v in sc.get("train/lr", []) if epoch_of(s) == ep]
            row.append(f"{lrs[-1]:.2e}" if lrs else "-")

            f.write("\t".join(row) + "\n")

    print(f"epoch 摘要 -> {out}")
    print()
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()