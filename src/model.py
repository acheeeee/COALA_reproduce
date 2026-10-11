"""COALA 完整模型（Stage 1）。"""
  
import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          WhisperModel)

from lora_setup import LM_ID, make_lora_config
from modules import AudioAdapter, CTCHead, compute_ctc_loss, ctc_compress
from prompt import build_stage1_sample, pad_batch

WHISPER_ID = "openai/whisper-large-v2"

CTC_WEIGHT = 0.3        # 論文 3.4 Stage 1
CE_WEIGHT = 1.0


class COALAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO 1: 載入 Whisper encoder（記得 dtype=torch.float32），
        #         .get_encoder()、.eval()、所有參數 requires_grad = False
        whisper = WhisperModel.from_pretrained(WHISPER_ID, dtype=torch.float32)
        self.encoder = whisper.get_encoder().eval()
        for p in self.encoder.parameters():
            p.requires_grad=False

        # TODO 2: self.adapter = AudioAdapter()
        #         self.ctc_head = CTCHead()

        self.adapter = AudioAdapter()
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