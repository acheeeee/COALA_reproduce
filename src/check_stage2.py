"""檢查 forward_stage2：KV cache 版必須和「每個候選各跑一次完整 forward」的結果一致。"""
import argparse
import random
import time

import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor

from biasing import add_unbiased, load_rare_pool, make_biasing_list
from dataset import LibriSpeechDataset
from model import COALAModel, WHISPER_ID
from modules import ctc_compress
from prompt import build_stage2_sample


@torch.no_grad()
def naive_scores(model, input_features, candidates):
    """對照組：不用 KV cache，每個候選把 [前綴 + entity] 拼成完整序列，各跑一次。"""
    h, ctc_logits = model.encode_audio(input_features)
    audio_tok = ctc_compress(h[0], ctc_logits[0])
    s2 = build_stage2_sample(audio_tok, candidates, model.tokenizer,
                             model.lm.get_input_embeddings())
    prefix = s2["prefix_embeds"]
    P = prefix.shape[0]
    s_list, logp_list = [], []
    for i in range(len(candidates)):
        n = int(s2["suffix_mask"][i].sum())
        seq = torch.cat([prefix, s2["suffix_embeds"][i, :n]], dim=0).unsqueeze(0)
        out = model.lm(inputs_embeds=seq, output_hidden_states=True)
        h_ent = out.hidden_states[-1][:, P:P + n]                    # (1, n, 576)
        s_list.append(model.projector(h_ent, torch.ones(1, n, device=seq.device))[0])
        logits = out.logits[0, P - 1:P + n - 1]                      # 位置 t 預測 t+1
        ids = s2["entity_ids"][i, :n]
        logp_list.append(-F.cross_entropy(logits, ids, reduction="sum"))
    return torch.stack(s_list), torch.stack(logp_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="Stage 1 checkpoint（不給就用隨機初始化的 adapter）")
    ap.add_argument("--stack", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(0)

    model = COALAModel(stack=args.stack)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        _, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected, f"checkpoint 有不認識的鍵: {unexpected}"
        print(f"載入 {args.ckpt}")
    model.setup_stage2()
    model.to(dev).eval()          # eval：關掉 LoRA dropout，兩條路才能逐位比較

    # 1. 可訓練參數
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[1] 可訓練參數 {n_tr:,}  (預期 332,929 + 921,600 = 1,254,529)")

    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)
    sample = ds[0]
    x = sample["input_features"].unsqueeze(0).to(dev)
    cands = ["<unbiased>", "quilter", "apostle", "gospel",
             "watson", "craswellers", "solved", "survey"]
    K = len(cands)

    # 2. shape
    out = model.forward_stage2(x, cands, need_logp=True)
    print(f"[2] s {tuple(out['s'].shape)}  logp {tuple(out['logp'].shape)}  (預期都是 ({K},))")

    # 3. KV cache 版 vs 完整版
    s_ref, logp_ref = naive_scores(model, x, cands)
    d_s = (out["s"] - s_ref).abs().max().item()
    d_l = (out["logp"] - logp_ref).abs().max().item()
    ok = d_s < 1e-3 and d_l < 1e-2
    print(f"[3] 分數最大差 {d_s:.2e}  logp 最大差 {d_l:.2e}  {'✅' if ok else '❌'}")
    for e, a, b, c in zip(cands, out["s"].tolist(), s_ref.tolist(), out["logp"].tolist()):
        print(f"    {e:<12} s={a:+.4f}  ref={b:+.4f}  logp={c:8.3f}")

    # 4. 梯度：只有 projector + LoRA scoring 應該拿到
    model.zero_grad(set_to_none=True)
    out = model.forward_stage2(x, cands, need_logp=True)
    (out["s"].sum() + out["logp"].sum()).backward()
    groups = {"projector": 0, "lora_scoring": 0, "lora_asr": 0, "adapter/ctc": 0, "其他": 0}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if n.startswith("projector."):
            groups["projector"] += 1
        elif ".scoring." in n:
            groups["lora_scoring"] += 1
        elif ".asr." in n:
            groups["lora_asr"] += 1
        elif n.startswith(("adapter.", "ctc_head.")):
            groups["adapter/ctc"] += 1
        else:
            groups["其他"] += 1
    print(f"[4] 拿到梯度的張量 {groups}")
    print(f"    (預期 projector 4、lora_scoring 120，其餘全 0)")

    # 5. 速度：用真正的 K = 121
    pool = load_rare_pool()
    pos = sample["rare_words"]
    full = add_unbiased(make_biasing_list(pos, pool, 120 - len(pos), random.Random(0)))
    with torch.no_grad():
        model.forward_stage2(x, full)                  # 暖機
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3):
            model.forward_stage2(x, full)
        torch.cuda.synchronize(); t_kv = (time.time() - t) / 3
        t = time.time()
        naive_scores(model, x, full)
        torch.cuda.synchronize(); t_nv = time.time() - t
    print(f"[5] K={len(full)}：KV cache {t_kv * 1000:.0f} ms / 句  vs  完整版 {t_nv * 1000:.0f} ms / 句"
          f"  → 快 {t_nv / t_kv:.1f} 倍")
    print(f"    GPU 峰值記憶體 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()