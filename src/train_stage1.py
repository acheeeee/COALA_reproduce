"""Stage 1 (ASR) 訓練腳本。

論文 3.4：10 epochs、batch size 4、CTC loss 權重 0.3 + CE 1.0。
論文未給 lr / optimizer / scheduler（疑點 #1），以下自行選定並記錄。
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

from biasing import load_rare_pool, make_stage1_list
from dataset import LibriSpeechDataset, collate_fn
from eval_utils import AMP_CHOICES, resolve_amp
from model import COALAModel, WHISPER_ID


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", type=int, default=2)
    ap.add_argument("--train", default="data/manifest/train-clean-100.tsv")
    ap.add_argument("--val", default="data/manifest/dev-clean.tsv")
    ap.add_argument("--out", default="exp/stage1_tc100")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", default="bf16", choices=AMP_CHOICES,
                    help="凍結 encoder 的精度；2080 Ti 沒有原生 bf16，用 auto 或 fp16")
    ap.add_argument("--early-stop-patience", type=int, default=10,
                    help="early stopping：epoch 結束時的 val loss 連續幾個 epoch 沒進步就停（官方預設 10，Stage 1 / 2 都不會觸發；續跑時計數歸零）")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def make_entity_lists(rare_words_batch, rare_pool, rng):
    return [make_stage1_list(rw, rare_pool, rng) for rw in rare_words_batch]


def to_device(batch, dev):
    for k in ["input_features", "ctc_ids", "ctc_lens"]:
        batch[k] = batch[k].to(dev, non_blocking=True)
    return batch


def save_ckpt(path, model, opt, sched, step, epoch, args):
    """只存可訓練參數，凍結的 encoder / LM 不用存（省 2.5GB）。"""
    trainable = {k: v for k, v in model.state_dict().items()
                 if k in {n for n, p in model.named_parameters() if p.requires_grad}}
    torch.save({
        "model": trainable,
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "epoch": epoch,
        "args": vars(args),
    }, path)


@torch.no_grad()
def validate(model, val_dl, rare_pool, dev, max_batches=50):
    """算 dev 上的平均 loss（WER 另外做，需要 generate）。"""
    model.eval()
    rng = random.Random(0)          # 固定 seed，每次驗證用同一組 biasing list
    tot, tot_ctc, tot_ce, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(val_dl):
        if i >= max_batches:
            break
        batch = to_device(batch, dev)
        el = make_entity_lists(batch["rare_words"], rare_pool, rng)
        out = model.forward_stage1(batch, el)
        tot += out["loss"].item()
        tot_ctc += out["ctc_loss"].item()
        tot_ce += out["ce_loss"].item()
        n += 1
    model.train()
    model.encoder.eval()
    return tot / n, tot_ctc / n, tot_ce / n


def main():
      args = parse_args()
      torch.manual_seed(args.seed)
      dev = torch.device("cuda")

      out_dir = Path(args.out)
      out_dir.mkdir(parents=True, exist_ok=True)
      writer = SummaryWriter(str(out_dir))
      print(f"輸出目錄: {out_dir}")

      # 每個 epoch 一行的摘要檔（開會用，可直接貼進 Excel）
      summary_path = out_dir / "epoch_summary.tsv"
      if not args.resume:
          with open(summary_path, "w", encoding="utf-8") as f:
              f.write("epoch\tstep\ttr_total\ttr_ctc\ttr_ce\t"
                      "val_total\tval_ctc\tval_ce\tlr\tmin\n")

      # --- 資料 ---
      fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
      train_ds = LibriSpeechDataset(args.train, fe)
      val_ds = LibriSpeechDataset(args.val, fe)
      train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True)
      val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True)

      rare_pool = load_rare_pool()
      rng = random.Random(args.seed)

      # --- 模型 ---
      model = COALAModel(stack=args.stack)
      model.amp_dtype = resolve_amp(args.amp)
      model.to(dev)
      model.train()
      model.encoder.eval()

      params = [p for p in model.parameters() if p.requires_grad]
      opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

      total_steps = len(train_dl) * args.epochs
      sched = get_linear_schedule_with_warmup(
          opt, int(total_steps * args.warmup_ratio), total_steps)

      print(f"stack = {args.stack}  (每秒 {1500 / args.stack / 30:.1f} 個位置)，encoder 精度 {model.amp_dtype}")
      print(f"可訓練參數 {sum(p.numel() for p in params):,}")
      print(f"總步數 {total_steps:,} ({len(train_dl)} 步/epoch × {args.epochs} epochs)",
            flush=True)

      start_epoch, step = 0, 0
      if args.resume:
          ck = torch.load(args.resume, map_location=dev, weights_only=False)
          missing, unexpected = model.load_state_dict(ck["model"], strict=False)
          assert len(unexpected) == 0, f"checkpoint 有不認識的鍵: {unexpected}"
          opt.load_state_dict(ck["optimizer"])
          sched.load_state_dict(ck["scheduler"])
          start_epoch, step = ck["epoch"], ck["step"]
          print(f"從 {args.resume} 續跑（epoch {start_epoch}, step {step}），"
                f"跳過 {len(missing)} 個凍結參數的鍵")

      # --- 訓練迴圈 ---
      t_start = time.time()
      best_val = float("inf")
      es_best, bad_epochs = float("inf"), 0      # early stopping（官方：看 epoch 結束的 valid/loss）

      for epoch in range(start_epoch, args.epochs):
          ep_tot = ep_ctc = ep_ce = 0.0
          ep_n = 0

          pbar = tqdm(train_dl, desc=f"epoch {epoch}", mininterval=30)
          for batch in pbar:
              step += 1

              batch = to_device(batch, dev)
              el = make_entity_lists(batch["rare_words"], rare_pool, rng)

              opt.zero_grad()
              out = model.forward_stage1(batch, el)
              out["loss"].backward()
              torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
              opt.step()
              sched.step()

              # epoch 平均用的累加器
              ep_tot += out["loss"].item()
              ep_ctc += out["ctc_loss"].item()
              ep_ce += out["ce_loss"].item()
              ep_n += 1

              if step % args.log_every == 0:
                  writer.add_scalar("train/total_loss", out["loss"].item(), step)
                  writer.add_scalar("train/ctc_loss", out["ctc_loss"].item(), step)
                  writer.add_scalar("train/ce_loss", out["ce_loss"].item(), step)
                  writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
                  pbar.set_postfix(
                      loss=out["loss"].item(),
                      ctc=out["ctc_loss"].item(),
                      ce=out["ce_loss"].item(),
                  )

              if step % args.val_every == 0:
                  v_tot, v_ctc, v_ce = validate(model, val_dl, rare_pool, dev)
                  writer.add_scalar("val/total_loss", v_tot, step)
                  writer.add_scalar("val/ctc_loss", v_ctc, step)
                  writer.add_scalar("val/ce_loss", v_ce, step)
                  if v_tot < best_val:
                      best_val = v_tot
                      save_ckpt(out_dir / "best.pt", model, opt, sched,
                                step, epoch, args)
                      print(f"\n[Step {step}] New best val loss: {best_val:.4f}",
                            flush=True)

          # --- epoch 結束：驗證 + 寫摘要 + 存檔 ---
          v_tot, v_ctc, v_ce = validate(model, val_dl, rare_pool, dev)
          mins = (time.time() - t_start) / 60

          line = (f"{epoch}\t{step}\t"
                  f"{ep_tot / ep_n:.4f}\t{ep_ctc / ep_n:.4f}\t{ep_ce / ep_n:.4f}\t"
                  f"{v_tot:.4f}\t{v_ctc:.4f}\t{v_ce:.4f}\t"
                  f"{sched.get_last_lr()[0]:.2e}\t{mins:.1f}")
          with open(summary_path, "a", encoding="utf-8") as f:
              f.write(line + "\n")
          print(f"\n[epoch {epoch} 摘要] tr_total={ep_tot / ep_n:.4f} "
                f"tr_ctc={ep_ctc / ep_n:.4f} tr_ce={ep_ce / ep_n:.4f} | "
                f"val_total={v_tot:.4f} val_ctc={v_ctc:.4f} val_ce={v_ce:.4f} | "
                f"{mins:.1f} 分", flush=True)

          writer.add_scalar("epoch/train_total", ep_tot / ep_n, epoch)
          writer.add_scalar("epoch/train_ctc", ep_ctc / ep_n, epoch)
          writer.add_scalar("epoch/train_ce", ep_ce / ep_n, epoch)
          writer.add_scalar("epoch/val_total", v_tot, epoch)
          writer.add_scalar("epoch/val_ctc", v_ctc, epoch)
          writer.add_scalar("epoch/val_ce", v_ce, epoch)

          if v_tot < best_val:
              best_val = v_tot
              save_ckpt(out_dir / "best.pt", model, opt, sched, step, epoch, args)
              print(f"[epoch {epoch}] New best val loss: {best_val:.4f}", flush=True)

          save_ckpt(out_dir / f"epoch{epoch}.pt", model, opt, sched,
                    step, epoch + 1, args)

          # --- early stopping：和官方相同，看 epoch 結束時的 val loss、min_delta = 0 ---
          if v_tot < es_best:
              es_best, bad_epochs = v_tot, 0
          else:
              bad_epochs += 1
              print(f"[early stop] val_total 已 {bad_epochs}/{args.early_stop_patience} 個 epoch 沒進步",
                    flush=True)
              if bad_epochs >= args.early_stop_patience:
                  print(f"\n[early stop] 停在 epoch {epoch}（最佳 val_total {es_best:.4f}）", flush=True)
                  break

      writer.close()
      print(f"\n完成。摘要檔: {summary_path}")

if __name__ == "__main__":
    main()