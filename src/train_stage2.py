"""Stage 2（Scoring）訓練腳本。

論文 3.4：凍結 Stage 1 全部參數，只訓練新的 projector + LoRA scoring；
7 epochs、batch size 1、M = 120 個候選（+ <unbiased> = 121）。
論文未給 lr / optimizer / scheduler；官方 code 用 AdamW lr 1e-4、grad accumulation 4。
"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, get_linear_schedule_with_warmup

from biasing import load_rare_pool, make_stage2_candidates
from dataset import LibriSpeechDataset, collate_fn
from losses import bias_loss, disc_loss, dpd_loss, mpd_loss
from model import COALAModel, WHISPER_ID
from train_stage1 import save_ckpt

LOSSES = {"disc": disc_loss, "mpd": mpd_loss, "dpd": dpd_loss}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True)
    ap.add_argument("--loss", required=True, choices=["disc", "bias", "mpd", "dpd"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--train", default="data/manifest/train-clean-100.tsv")
    ap.add_argument("--val", default="data/manifest/dev-clean.tsv")
    ap.add_argument("--epochs", type=int, default=7)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--val-utts", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=None, help="只跑 N 句就停（量速度用）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resume", default=None, help="從 epoch{N}.pt 續跑")
    return ap.parse_args()


def compute_loss(name, out, pos):
    """依 --loss 選擇目標函數。out = forward_stage2 的回傳，pos = (K,) bool。"""
    if name == "bias":
        return bias_loss(out["s"], pos, out["logp"])
    return LOSSES[name](out["s"], pos)

@torch.no_grad()
def validate(model, val_ds, rare_pool, loss_name, dev, n_utts):
    """dev-clean 前 n_utts 句：平均 loss + Recall#20（不含 <unbiased>）。"""
    model.eval()
    rng = random.Random(0)                    # 每次驗證用同一組干擾詞，數字才能互相比
    tot_loss, hit, n_pos = 0.0, 0, 0
    n = min(n_utts, len(val_ds))
    for i in range(n):
        smp = val_ds[i]
        full, pos = make_stage2_candidates(smp["rare_words"], rare_pool, rng)
        pos = pos.to(dev)
        x = smp["input_features"].unsqueeze(0).to(dev)
        out = model.forward_stage2(x, full, need_logp=(loss_name == "bias"))
        tot_loss += compute_loss(loss_name, out, pos).item()

        real = pos.clone()
        real[0] = False                       # 只算真的 rare word
        if real.any():
            top = out["s"].topk(20).indices
            hit += int(real[top].sum())
            n_pos += int(real.sum())
    model.train()
    model.encoder.eval()
    return tot_loss / n, hit / max(n_pos, 1)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir))
    summary_path = out_dir / "epoch_summary.tsv"

    if not args.resume:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("epoch\tstep\ttr_loss\tval_loss\tval_recall20\tlr\tmin\n")
    # --- 資料 ---
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    train_ds = LibriSpeechDataset(args.train, fe)
    val_ds = LibriSpeechDataset(args.val, fe)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn,
                          num_workers=args.num_workers, pin_memory=True)
    rare_pool = load_rare_pool()
    rng = random.Random(args.seed)

    # --- 模型：載入 Stage 1 → 切到 Stage 2 ---
    model = COALAModel(stack=args.stack)
    ck = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"checkpoint 有不認識的鍵: {unexpected}"
    model.setup_stage2()
    model.to(dev)
    model.train()
    model.encoder.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    n_updates = len(train_dl) * args.epochs // args.grad_accum
    sched = get_linear_schedule_with_warmup(opt, int(n_updates * args.warmup_ratio), n_updates)
    need_logp = args.loss == "bias"
    start_epoch, step = 0, 0
    if args.resume:
        rk = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(rk["model"], strict=False)
        opt.load_state_dict(rk["optimizer"])
        sched.load_state_dict(rk["scheduler"])
        start_epoch, step = rk["epoch"], rk["step"]
        print(f"從 {args.resume} 續跑：epoch {start_epoch}、step {step}", flush=True)
    print(f"loss = {args.loss}  可訓練參數 {sum(p.numel() for p in params):,}")
    print(f"{len(train_dl)} 句/epoch × {args.epochs} epochs，每 {args.grad_accum} 句更新一次"
          f" → {n_updates:,} 次更新", flush=True)

    # --- 訓練迴圈 ---
    best_recall, t0 = -1.0, time.time()        # step 不再這裡歸零
    opt.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        ep_loss, ep_n = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", mininterval=30)
        for batch in pbar:
            step += 1
            x = batch["input_features"].to(dev, non_blocking=True)        # (1, 80, 3000)
            full, pos = make_stage2_candidates(batch["rare_words"][0], rare_pool, rng)
            pos = pos.to(dev)

            out = model.forward_stage2(x, full, need_logp=need_logp)
            loss = compute_loss(args.loss, out, pos)
            (loss / args.grad_accum).backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()

            ep_loss += loss.item()
            ep_n += 1

            if step % args.log_every == 0:
                s = out["s"].detach()
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                writer.add_scalar("train/pos_score_mean", s[pos].mean().item(), step)
                writer.add_scalar("train/neg_score_mean", s[~pos].mean().item(), step)
                writer.add_scalar("train/unbiased_score", s[0].item(), step)
                pbar.set_postfix(loss=loss.item())

            if step == 100:
                sec = (time.time() - t0) / 100
                total = len(train_dl) * args.epochs
                print(f"\n[速度] {sec:.2f} 秒/句 → 全部 {total:,} 句約 {sec * total / 3600:.1f} 小時"
                      f"，GPU 峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)

            if args.max_steps and step >= args.max_steps:
                print(f"\n到達 --max-steps {args.max_steps}，停止。最後 loss = {loss.item():.4f}")
                writer.close()
                return

            if step % args.val_every == 0:
                v_loss, v_r20 = validate(model, val_ds, rare_pool, args.loss, dev, args.val_utts)
                writer.add_scalar("val/loss", v_loss, step)
                writer.add_scalar("val/recall20", v_r20, step)
                if v_r20 > best_recall:
                    best_recall = v_r20
                    save_ckpt(out_dir / "best.pt", model, opt, sched, step, epoch, args)
                    print(f"\n[step {step}] new best Recall#20 = {v_r20:.4f}", flush=True)

        # --- epoch 結束：驗證 + 摘要 + 存檔 ---
        v_loss, v_r20 = validate(model, val_ds, rare_pool, args.loss, dev, args.val_utts)
        mins = (time.time() - t0) / 60
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch}\t{step}\t{ep_loss / ep_n:.4f}\t{v_loss:.4f}\t{v_r20:.4f}\t"
                    f"{sched.get_last_lr()[0]:.2e}\t{mins:.1f}\n")
        print(f"\n[epoch {epoch}] tr_loss={ep_loss / ep_n:.4f} val_loss={v_loss:.4f} "
              f"val_R#20={v_r20:.4f} | {mins:.1f} 分", flush=True)
        writer.add_scalar("epoch/train_loss", ep_loss / ep_n, epoch)
        writer.add_scalar("epoch/val_loss", v_loss, epoch)
        writer.add_scalar("epoch/val_recall20", v_r20, epoch)
        if v_r20 > best_recall:
            best_recall = v_r20
            save_ckpt(out_dir / "best.pt", model, opt, sched, step, epoch, args)
        save_ckpt(out_dir / f"epoch{epoch}.pt", model, opt, sched, step, epoch + 1, args)

    writer.close()
    print(f"\n完成。最佳 val Recall#20 = {best_recall:.4f}，摘要檔: {summary_path}")


if __name__ == "__main__":
    main()