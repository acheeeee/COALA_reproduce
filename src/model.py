"""COALA 完整模型（Stage 1）。"""
  
import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          WhisperModel)

from lora_setup import LM_ID, make_lora_config
from modules import AudioAdapter, CTCHead, compute_ctc_loss, ctc_compress
from prompt import build_stage1_sample, pad_batch

import torch.nn.functional as F
from projector import DiscriminativeProjector
from prompt import IGNORE_INDEX, build_stage2_sample

WHISPER_ID = "openai/whisper-large-v2"

CTC_WEIGHT = 0.3        # 論文 3.4 Stage 1
CE_WEIGHT = 1.0


class COALAModel(nn.Module):
    def __init__(self, stack=2):
        super().__init__()

        # TODO 1: 載入 Whisper encoder（記得 dtype=torch.float32），
        #         .get_encoder()、.eval()、所有參數 requires_grad = False
        whisper = WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32)
        self.encoder = whisper.get_encoder().eval()
        for p in self.encoder.parameters():
            p.requires_grad=False

        # TODO 2: self.adapter = AudioAdapter()
        #         self.ctc_head = CTCHead()

        self.adapter = AudioAdapter(stack=stack)
        self.ctc_head = CTCHead()

        # TODO 3: 載入 lm（dtype=torch.float32），凍結它的全部參數，
        #         再用 get_peft_model(lm, make_lora_config(), adapter_name="asr") 包起來
        #         加上 self.lm.add_adapter("scoring", make_lora_config())
        self.lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
        for n in self.lm.parameters():
            n.requires_grad=False
        self.lm = get_peft_model(self.lm, make_lora_config(), adapter_name="asr")
        self.lm.add_adapter("scoring", make_lora_config())

        # TODO 4: self.tokenizer = AutoTokenizer.from_pretrained(LM_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(LM_ID)

    def encode_audio(self, input_features):
        """(B, 80, 3000) → h (B, 375, 576), ctc_logits (B, 375, 29)"""
        # TODO 5: encoder 用 with torch.no_grad()（它是凍結的，不用算梯度）
        #         然後 adapter、ctc_head
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.encoder(input_features)
        x = out.last_hidden_state.float()
        h = self.adapter(x)
        ctc_logits = self.ctc_head(h)
        return h, ctc_logits

    def forward_stage1(self, batch, entity_lists):
        """batch 來自 dataset 的 collate_fn，entity_lists 是每句的 10 個 entity。"""
        feats = batch["input_features"]
        h, ctc_logits = self.encode_audio(feats)

        # TODO 6: ctc_loss = compute_ctc_loss(ctc_logits, batch["ctc_ids"],
        #                                     batch["ctc_lens"])
        ctc_loss = compute_ctc_loss(ctc_logits, batch["ctc_ids"], batch["ctc_lens"])

        # TODO 7: 逐句做 ctc_compress + build_stage1_sample，收集成 list
        #         embed_layer = self.lm.get_input_embeddings()
        #         for i in range(len(batch["text"])):
        #             audio_tok = ctc_compress(h[i], ctc_logits[i])
        #             samples.append(build_stage1_sample(
        #                 audio_tok, entity_lists[i], batch["text"][i],
        #                 self.tokenizer, embed_layer))
        samples = []
        embed_layer = self.lm.get_input_embeddings()
        for i in range(len(batch["text"])):
            audio_tok = ctc_compress(h[i], ctc_logits[i])
            samples.append(build_stage1_sample(
                audio_tok, entity_lists[i], batch["text"][i],
                         self.tokenizer, embed_layer))

        # TODO 8: padded = pad_batch(samples)
        #         out = self.lm(**padded)       ← HF 會自己用 labels 算 CE loss
        #         ce_loss = out.loss
        padded = pad_batch(samples)
        out = self.lm(**padded)       
        ce_loss = out.loss
        # TODO 9: total = CTC_WEIGHT * ctc_loss + CE_WEIGHT * ce_loss
        #         return dict(loss=total, ctc_loss=ctc_loss, ce_loss=ce_loss)
        total = CTC_WEIGHT * ctc_loss + CE_WEIGHT * ce_loss
        return dict(loss=total, ctc_loss=ctc_loss, ce_loss=ce_loss)

    def setup_stage2(self):
        """Stage 2 準備：新建 projector，凍結 Stage 1 的一切，只訓練 projector + LoRA scoring。

        呼叫順序：建模型 → 載入 Stage 1 checkpoint → setup_stage2() → .to(dev)
        （在 .to(dev) 之前建 projector，它才會一起搬上 GPU）
        """
        self.projector = DiscriminativeProjector()

        for m in ( self.adapter, self.ctc_head):          # 兩個要凍結的模組
            for p in m.parameters():          # 逐個取出參數
                p.requires_grad = False          # 標成不可訓練

        self.lm.set_adapter("scoring")

    def forward_stage2(self, input_features, candidates, need_logp=False):
        """
        一句話 + K 個候選 → K 個分數 s（+ 可選的 log P(e_i | X)，給 bias_loss 用）

        input_features: (1, 80, 3000)
        candidates:     list[str]，長度 K，index 0 = <unbiased>
        回傳 dict:      s (K,)、logp (K,) 或 None
        """
        # ── 0. 音訊 → audio tokens（encoder / adapter / CTC 都凍結，這裡用 no_grad 沒問題）──
        with torch.no_grad():
            h, ctc_logits = self.encode_audio(input_features)
            audio_tok = ctc_compress(h[0], ctc_logits[0])

        embed_layer = self.lm.get_input_embeddings()
        s2 = build_stage2_sample(audio_tok, candidates, self.tokenizer, embed_layer)
        prefix, suffix = s2["prefix_embeds"], s2["suffix_embeds"]
        suffix_mask, entity_ids = s2["suffix_mask"], s2["entity_ids"]
        K, P = suffix.shape[0], prefix.shape[0]

        out_p = self.lm(inputs_embeds=prefix.unsqueeze(0), use_cache=True)
        cache = out_p.past_key_values


        cache.batch_repeat_interleave(K)

        prefix_mask = torch.ones(K, P, dtype=suffix_mask.dtype, device=suffix_mask.device)
        attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)

        out_s = self.lm(
        inputs_embeds=suffix,
        attention_mask=attention_mask,
        past_key_values=cache,
        output_hidden_states=True,
        )

        h_ent = out_s.hidden_states[-1] 
        s = self.projector(h_ent, suffix_mask.float())

        logp = None
        logp = None
        if need_logp:
            # 預測「第 1 個 token」:用前綴最後一個位置的輸出,複製成 K 份
            logits_first = out_p.logits[:, -1:].expand(K, -1, -1)        # (K, 1, V)
            # 預測「第 2 個以後」:用尾巴位置 0 ~ S-2 的輸出
            logits_rest = out_s.logits[:, :-1]                           # (K, S-1, V)
            # 接起來,剛好和 entity_ids (K, S) 逐格對齊
            logits = torch.cat([logits_first, logits_rest], dim=1)       # (K, S, V)
            # 每個位置的 -log 機率,補位(-100)的位置是 0
            ce = F.cross_entropy(logits.transpose(1, 2), entity_ids,
                                 ignore_index=IGNORE_INDEX, reduction="none")   # (K, S)
            # 每個候選:加總再變號
            logp = -ce.sum(dim=1)                                        # (K,)

        return {"s": s, "logp": logp}
        return {"s": s, "logp": logp}


def main():
    import random
    from torch.utils.data import DataLoader
    from transformers import WhisperFeatureExtractor

    from biasing import load_rare_pool, make_stage1_list
    from dataset import LibriSpeechDataset, collate_fn

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    # --- 資料 ---
    fe = WhisperFeatureExtractor.from_pretrained(WHISPER_ID)
    ds = LibriSpeechDataset("data/manifest/dev-clean.tsv", fe)
    dl = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))

    # --- 每句產生 10 個 entity 的 biasing list（論文 Stage 1 設定）---
    rare_pool = load_rare_pool()
    rng = random.Random(42)
    entity_lists = [
        make_stage1_list(rw, rare_pool, rng) for rw in batch["rare_words"]
    ]
    for utt, rw, el in zip(batch["utt_id"], batch["rare_words"], entity_lists):
        print(f"  {utt}: E+ {rw} -> 清單 {len(el)} 個")

    # --- 模型 ---
    model = COALAModel().to(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可訓練參數: {n_train:,}  (預算 5,700,000)")

    for k in ["input_features", "ctc_ids", "ctc_lens"]:
        batch[k] = batch[k].to(dev)

    out = model.forward_stage1(batch, entity_lists)
    ln_vocab = torch.log(torch.tensor(49152.0)).item()
    print(f"\nctc_loss = {out['ctc_loss'].item():.4f}   (基準 ~10)")
    print(f"ce_loss  = {out['ce_loss'].item():.4f}   (基準 ln(49152)={ln_vocab:.2f})")
    print(f"total    = {out['loss'].item():.4f}")

    # --- 確認梯度能回傳到所有可訓練參數 ---
    out["loss"].backward()
    n_grad = sum(1 for p in model.parameters()
                 if p.requires_grad and p.grad is not None)
    n_tr = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"\n有梯度的張量: {n_grad}/{n_tr}")


if __name__ == "__main__":
    main()