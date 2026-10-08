"""Stage 1 (ASR) 的 prompt 組裝。

序列結構（四段）：
    [seg1 文字] [audio tokens] [seg2 文字] [target 文字]
    seg1 = <|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n
    seg2 = {instruction}<|im_end|>\n<|im_start|>assistant\n
    target = {逐字稿}<|im_end|>

    labels: seg1 / audio / seg2 全部 -100，只有 target 放真實 token ids

論文 3.4 Stage 1：prompt 含最多 5 個正樣本、共 10 個 entity。
論文未給完整 instruction 文字（疑點 #7），以下自行設計並記錄。
"""

import torch

IGNORE_INDEX = -100

SYSTEM_PROMPT = "You are a speech recognition assistant."

INSTRUCTION_TEMPLATE = (
    "Transcribe the speech. "
    "The following entities may appear: {entities}."
)

#bos_token -> beginning of sequqnce
#eos_token -> end of sequence

def build_seg1(tokenizer):
    """chat template 的開頭到 user turn 開始。"""
    return (
        f"{tokenizer.bos_token}system\n{SYSTEM_PROMPT}{tokenizer.eos_token}\n"
        f"{tokenizer.bos_token}user\n"
    )


def build_seg2(entity_list, tokenizer):
    """instruction + entities，接到 assistant turn 開始。"""
    instruction = INSTRUCTION_TEMPLATE.format(entities=", ".join(entity_list))  
    # .join(list) -> 把list 用 ", "的方式接成一個字串
    # .format(entities) 就是把entities的句子接上 INSTRUCTION_TEMPLATE

    return (
        f"{instruction}{tokenizer.eos_token}\n"
        f"{tokenizer.bos_token}assistant\n"
    )


def encode(text, tokenizer, device):
    """文字 → token id tensor，不加 special tokens（template 手動控制）。"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # tokenizer(text) 會把text切成token 然後換成 id 最後會像dict 所以只取 "input_ids"
                        #add_special_tokens=False 因為bos, eos已經手動加就不用再重複
    return torch.tensor(ids, dtype=torch.long, device=device) #把 ids 放到指定的 device


def build_stage1_sample(audio_tokens, entity_list, target_text, tokenizer, embed_layer):
    """組出單句的 inputs_embeds / labels / attention_mask。

    Args:
        audio_tokens: (L, 576)  ctc_compress 的輸出
        entity_list:  list[str] 10 個 entity
        target_text:  str       逐字稿（小寫）
        tokenizer:    AutoTokenizer
        embed_layer:  lm.get_input_embeddings()
    """
    dev = audio_tokens.device

    # TODO 1: 用 build_seg1 / build_seg2 組出兩段文字，
    #         target 文字 = target_text + tokenizer.eos_token
    seg1 = build_seg1(tokenizer) 
    seg2 = build_seg2(entity_list, tokenizer)
    target = target_text + tokenizer.eos_token

    # TODO 2: 三段文字各用 encode() 轉成 id tensor
    seg1_ids    = encode(seg1, tokenizer, dev)
    seg2_ids    = encode(seg2, tokenizer, dev)
    target_ids  = encode(target, tokenizer, dev)

    # TODO 3: 三段 id 各用 embed_layer(ids) 查成 (n, 576)
    seg1_emb = embed_layer(seg1_ids)
    seg2_emb = embed_layer(seg2_ids)
    target_emb = embed_layer(target_ids)

    # TODO 4: torch.cat 四段 → inputs_embeds (L+n1+n2+m, 576)
    #         順序：seg1, audio_tokens, seg2, target
    #eg     a shape (2, 3)
    #       b shape (4, 3)
    #       torch.cat([a, b], dim=0)   →  shape (6, 3)    # 第 0 維:2+4=6
    inputs_embeds = torch.cat([seg1_emb, audio_tokens, seg2_emb, target_emb], dim=0)

    # TODO 5: labels = cat([
    #             torch.full((n1 + L + n2,), IGNORE_INDEX,
    #                        dtype=torch.long, device=dev),
    #             target_ids,
    #         ])
    n1 = seg1_emb.shape[0]
    n2 = seg2_emb.shape[0]
    L = len(audio_tokens)
    m = target_emb.shape[0]
    labels = torch.cat([
                torch.full((n1 + L + n2,), IGNORE_INDEX,
                            dtype=torch.long, device=dev),
                target_ids,
            ])
    # TODO 6: attention_mask = torch.ones(總長, dtype=torch.long, device=dev)
    attention_mask = torch.ones(n1+L+n2+m, dtype=torch.long, device=dev)

    return {
        "inputs_embeds": inputs_embeds,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(MID)
    lm = AutoModelForCausalLM.from_pretrained(MID)
    embed = lm.get_input_embeddings()

    audio = torch.randn(89, 576)
    entities = ["apostle", "gospel", "quilter", "foo", "bar",
                "baz", "qux", "quux", "corge", "grault"]
    text = "mister quilter is the apostle of the middle classes"

    s = build_stage1_sample(audio, entities, text, tok, embed)

    print(f"inputs_embeds  {tuple(s['inputs_embeds'].shape)}")
    print(f"labels         {tuple(s['labels'].shape)}")
    print(f"attention_mask {tuple(s['attention_mask'].shape)}")

    n_loss = (s["labels"] != IGNORE_INDEX).sum().item()
    n_target = len(tok(text + tok.eos_token, add_special_tokens=False)["input_ids"])
    print(f"\n算 loss 的位置: {n_loss}  (target token 數 = {n_target})")

    print(f"\n--- seg1 ---\n{build_seg1(tok)}")
    print(f"--- seg2 ---\n{build_seg2(entities, tok)}")
    print(
        f"文字 token 總數: "
        f"{len(tok(build_seg1(tok), add_special_tokens=False)['input_ids']) + len(tok(build_seg2(entities, tok), add_special_tokens=False)['input_ids'])}"
    )


if __name__ == "__main__":
    main()