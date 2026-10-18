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

INSTRUCTION_NO_BIAS = "Transcribe the speech."

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

    if entity_list:
        instruction = INSTRUCTION_TEMPLATE.format(
            entities=", ".join(entity_list))
    else:
        instruction = INSTRUCTION_NO_BIAS

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

def pad_batch(samples, pad_side="right"):
    """把 build_stage1_sample 的多個輸出 pad 成 batch。

    Args:
        samples: list of dict，各含 inputs_embeds (tot_i, 576) /
                 labels (tot_i,) / attention_mask (tot_i,)
        pad_side: "right" 訓練用、"left" 生成用

    Returns:
        dict 含 (B, max_tot, 576) / (B, max_tot) / (B, max_tot)
    """
    B = len(samples)
    lens = [s["inputs_embeds"].shape[0] for s in samples]
    max_len = max(lens)
    d = samples[0]["inputs_embeds"].shape[1]
    dev = samples[0]["inputs_embeds"].device
    dtype = samples[0]["inputs_embeds"].dtype

    embeds = torch.zeros(B, max_len, d, dtype=dtype, device=dev)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long, device=dev)
    mask = torch.zeros(B, max_len, dtype=torch.long, device=dev)

    for i, s in enumerate(samples):
        n = lens[i]
        if pad_side == "right":
            sl = slice(0, n)
        else:
            sl = slice(max_len - n, max_len)
        embeds[i, sl] = s["inputs_embeds"]
        labels[i, sl] = s["labels"]
        mask[i, sl] = s["attention_mask"]

    return {"inputs_embeds": embeds, "labels": labels, "attention_mask": mask}

def build_gen_prompt(audio_tokens, entity_list, tokenizer, embed_layer):
    """生成用的 prompt：只到 assistant turn 開始，不含 target。

    Returns:
        inputs_embeds (1, n1+L+n2, 576), attention_mask (1, n1+L+n2)
    """
    dev = audio_tokens.device

    # TODO 1: seg1 / seg2 用 build_seg1 / build_seg2

    seg1 = build_seg1(tokenizer)
    seg2 = build_seg2(entity_list,tokenizer)
    #seg1,seg2都是已經加好前後綴的文字句了

    # TODO 2: 兩段用 encode() 轉 id、再用 embed_layer 查成向量
    #   def encode(text, tokenizer, device)
    seg1_ids = encode(seg1, tokenizer, dev)
    seg2_ids = encode(seg2, tokenizer, dev)
    #seg1_ids,seg2_ids都是裝編號的tensor

    seg1_emb = embed_layer(seg1_ids)
    seg2_emb = embed_layer(seg2_ids)

    # TODO 3: torch.cat([seg1_emb, audio_tokens, seg2_emb], dim=0)
    #         然後 .unsqueeze(0) 加上 batch 維 → (1, total, 576)

    inputs_embeds = torch.cat([seg1_emb, audio_tokens, seg2_emb], dim=0).unsqueeze(0)
    total = inputs_embeds.shape[1]

    # TODO 4: attention_mask = torch.ones(1, total, dtype=torch.long, device=dev)
    attention_mask = torch.ones(1, total, dtype=torch.long, device=dev)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


STAGE2_INSTRUCTION = "Please determine if the text appears in the input speech. Text:"
# 論文圖 2(a) 只露出 "...Please determine if the text appears in the input speech"，其餘自訂

def build_stage2_sample(audio_tokens, candidates, tokenizer, embed_layer):
    """
    Stage 2：一句話 + K 個候選 → 一段共用前綴 + K 條尾巴（尾巴只放 entity）

    Args:
        audio_tokens: (L', 576)   ctc_compress 的輸出
        candidates:   list[str]   長度 K，index 0 是 <unbiased>（= add_unbiased 的輸出）
        tokenizer, embed_layer:   同 Stage 1

    Returns dict:
        prefix_embeds (P, 576)      seg1 + audio + 題目，K 條共用
        suffix_embeds (K, S, 576)   每條尾巴 = 一個 entity 的 token 向量，右側補 0
        suffix_mask   (K, S) long   1 = entity token，0 = pad
        entity_ids    (K, S) long   entity 的 token id，pad 位置填 IGNORE_INDEX
    """
    dev = audio_tokens.device

    # TODO(1) 前綴的兩段文字：seg1 用 build_seg1()，題目用 STAGE2_INSTRUCTION
    #         各自 encode() → embed_layer() 查表
    seg1 = build_seg1(tokenizer)
    seg1_ids = encode(seg1, tokenizer, dev)
    inst_ids = encode(STAGE2_INSTRUCTION, tokenizer, dev)
    seg1_emb = embed_layer(seg1_ids)
    inst_emb = embed_layer(inst_ids)

    # TODO(2) prefix_embeds = cat([seg1_emb, audio_tokens, inst_emb], dim=0)
    prefix_embeds = torch.cat([seg1_emb, audio_tokens, inst_emb], dim=0)
    
    # TODO(3) 每個候選各自 encode() → 得到 K 個長度不同的 id tensor
    #         要encode(" " + e)
    cand_ids = []
    for e in candidates:
        cand_ids.append(encode(" "+ e, tokenizer, dev))

    # TODO(4) K = 候選數、S = 最長的 token 數

    K = len(candidates)
    lens = []
    for n in cand_ids:
        lens.append(len(n))
    #等價 lens = [len(n) for n in cand_ids] 一行版本
    S = max(lens)
    suffix_embeds  = torch.zeros((K, S, 576), dtype=torch.float32, device=dev)
    suffix_mask    = torch.zeros((K, S), dtype=torch.long, device=dev)
    entity_ids     = torch.full((K, S), IGNORE_INDEX, dtype=torch.long, device=dev)


    # TODO(5) 逐個候選填進去：先用 embed_layer 查表，再放進第 i 列的前 n 格
    #         （參考 pad_batch() 的右 pad 寫法）
    for i, ids in enumerate(cand_ids):
        n = len(ids)                            # 第 i 個候選的 token 數,整數
        suffix_embeds[i, :n] = embed_layer(ids) # (n, 576),向量
        suffix_mask[i, :n] = 1                  # 標成「真的 token」
        entity_ids[i, :n] = ids                 # 這個候選的 token 編號

    return {
        "prefix_embeds": prefix_embeds,
        "suffix_embeds": suffix_embeds,
        "suffix_mask": suffix_mask,
        "entity_ids": entity_ids,
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
        # ── Stage 2 ──
    print("\n========== Stage 2 ==========")
    cands = ["<unbiased>", "watson", "solved", "craswellers"]
    s2 = build_stage2_sample(audio, cands, tok, embed)

    n_seg1 = len(tok(build_seg1(tok), add_special_tokens=False)["input_ids"])
    n_inst = len(tok(STAGE2_INSTRUCTION, add_special_tokens=False)["input_ids"])
    print(f"prefix_embeds  {tuple(s2['prefix_embeds'].shape)}  "
          f"(預期 ({n_seg1} + {audio.shape[0]} + {n_inst}, 576))")
    print(f"suffix_embeds  {tuple(s2['suffix_embeds'].shape)}")
    print(f"suffix_mask    {tuple(s2['suffix_mask'].shape)}")
    print(f"entity_ids     {tuple(s2['entity_ids'].shape)}")

    print("\n每個候選：")
    for i, e in enumerate(cands):
        m = s2["suffix_mask"][i].bool()
        ids = s2["entity_ids"][i][m]
        print(f"  {e:<12} {int(m.sum())} 個 token  decode={tok.decode(ids)!r}")

    # 斷詞邊界檢查：分開 encode 的結果，必須等於「題目 + entity」整句一起 encode
    print("\n斷詞邊界檢查：")
    for i, e in enumerate(cands):
        joint = tok(STAGE2_INSTRUCTION + " " + e, add_special_tokens=False)["input_ids"]
        inst = tok(STAGE2_INSTRUCTION, add_special_tokens=False)["input_ids"]
        m = s2["suffix_mask"][i].bool()
        mine = s2["entity_ids"][i][m].tolist()
        print(f"  {e:<12} {'✅' if joint == inst + mine else '❌'}  "
              f"整句={joint[len(inst):]}  分開={mine}")

if __name__ == "__main__":
    main()