"""在 SmolLM2 上掛兩組具名 LoRA adapter（ASR / Scoring）。

論文圖 2(a)：Backbone LM 內部有兩組 LoRA，
  Stage 1 訓練 ASR 那組；
  Stage 2 凍結全部舊參數、只訓練新的 Scoring 那組 + projector。

論文未給 LoRA 設定（疑點 #3）。以下 rank 由參數預算反推：
  可訓練預算 5.7M − adapter/CTC 2.97M − projector 0.33M ≈ 2.4M
  兩組 LoRA 各約 1.2M；SmolLM2 有 30 層、hidden 576
  → 每層每組約 r × 1920 → 30 層約 r × 57,600 → r ≈ 16~20
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

LM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

LORA_RANK = 16          # 上限維度 r=16
LORA_ALPHA = 32         # 縮放係數
LORA_DROPOUT = 0.05     # 隨機丟掉防止過擬合
TARGET_MODULES = ["q_proj", "v_proj"]


def make_lora_config():
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",                    # 不訓練本來的bias 因為要把它凍結起來
        task_type="CAUSAL_LM",
    )


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_active(model, names):
    """切換 active adapter。
    - str  : PeftModel.set_adapter 只吃單一字串
    - list : 要呼叫底層 LoraModel（model.base_model）的 set_adapter 才吃 list
    """
    if isinstance(names, str):
        model.set_adapter(names)
    else:
        model.base_model.set_adapter(list(names))


def freeze_adapter(model, name):
    """把名字裡含 .<name>. 的參數全部 requires_grad=False。"""
    for n, p in model.named_parameters():
        if f".{name}." in n:
            p.requires_grad = False


def which_trainable(model, tag):
    """看 requires_grad=True 的參數落在哪一組 adapter。"""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    n_asr = sum(1 for n in names if ".asr." in n)
    n_sco = sum(1 for n in names if ".scoring." in n)
    tot = count_trainable(model)
    print(f"[{tag}] 可訓練 {tot:,} | asr 張量 {n_asr} | scoring 張量 {n_sco} "
          f"| 其他 {len(names) - n_asr - n_sco}")
    print(f"        active_adapters = {model.active_adapters}")


def which_gets_grad(model, tag):
    """做一次 forward + backward，看哪一組的參數真的拿到 grad（grad 不是 None）。
    requires_grad=False 的參數 backward 後 grad 會是 None，
    所以這個測試能確認「實際在訓練的是誰」。"""
    model.zero_grad(set_to_none=True)
    x = torch.randn(1, 20, 576)
    loss = model(inputs_embeds=x).logits.pow(2).mean()
    loss.backward()
    names = [n for n, p in model.named_parameters() if p.grad is not None]
    n_asr = sum(1 for n in names if ".asr." in n)
    n_sco = sum(1 for n in names if ".scoring." in n)
    print(f"[{tag}] 拿到 grad: asr 張量 {n_asr} | scoring 張量 {n_sco} "
          f"| 其他 {len(names) - n_asr - n_sco}")
    model.zero_grad(set_to_none=True)


def main():
    # TODO 1
    lm = AutoModelForCausalLM.from_pretrained(LM_ID, dtype=torch.float32)
    print(f"total parameter: {count_trainable(lm)/1e6:.1f}M")  # 1e6 = 1,000,000 . :.1f 是小數點後留一位

    # TODO 2
    model = get_peft_model(lm, make_lora_config(), adapter_name="asr")
    model.print_trainable_parameters()

    # TODO 3
    model.add_adapter("scoring", make_lora_config())
    model.print_trainable_parameters()

    # TODO 4
    set_active(model, "asr")
    print("ASR: ")
    model.print_trainable_parameters()
    set_active(model, "scoring")
    print("Scoring: ")
    model.print_trainable_parameters()

    # TODO 5
    x = torch.randn(1, 20, 576)
    out = model(inputs_embeds=x)
    print(out.logits.shape)

    # --- 1. 單組啟用時，requires_grad 到底落在誰身上？---
    print("\n=== 1. 單組啟用：requires_grad 落在誰身上 ===")
    set_active(model, "asr")
    which_trainable(model, "set_adapter('asr')")
    set_active(model, "scoring")
    which_trainable(model, "set_adapter('scoring')")

    # --- 2. 兩組同時啟用（改用 base_model.set_adapter）---
    print("\n=== 2. 兩組同時啟用 ===")
    try:
        set_active(model, ["asr", "scoring"])
        which_trainable(model, "both active")
    except Exception as e:
        print(f"兩組同時啟用失敗: {type(e).__name__}: {e}")

    # --- 2b. Stage 2 情境：兩組都啟用，但 asr 凍結、只訓練 scoring ---
    print("\n=== 2b. Stage 2：both active + asr frozen ===")
    freeze_adapter(model, "asr")
    which_trainable(model, "both active, asr frozen")

    # --- 3. 實際做 backward，看誰真的拿到 grad ---
    # 這個比數參數量更能分辨「切對了沒」，因為兩組 config 一樣，參數量相同
    print("\n=== 3. backward 後誰拿到 grad ===")
    model.eval()  # 關掉 lora_dropout，避免隨機性

    # lora_B 預設是 0，要先給非零值，grad 與 delta 才有意義
    # asr 和 scoring 用不同 seed，這樣兩組的輸出才分辨得開
    for name, seed in [("asr", 0), ("scoring", 1)]:
        g = torch.Generator().manual_seed(seed)
        for n, p in model.named_parameters():
            if f".{name}." in n and "lora_B" in n:
                with torch.no_grad():
                    p.copy_(torch.randn(p.shape, generator=g) * 0.02)

    set_active(model, "asr")
    which_trainable(model, "single: asr")
    which_gets_grad(model, "single: asr")

    set_active(model, "scoring")
    which_trainable(model, "single: scoring")
    which_gets_grad(model, "single: scoring")

    # Stage 2 情境：asr 要凍結、scoring 訓練
    set_active(model, ["asr", "scoring"])
    freeze_adapter(model, "asr")
    which_trainable(model, "both active, asr frozen")
    which_gets_grad(model, "both active, asr frozen")

    # --- 4. 兩組的 delta 真的有相加嗎？---
    print("\n=== 4. logits 差異（delta 是否生效）===")
    x = torch.randn(1, 20, 576)
    with torch.no_grad():
        set_active(model, "asr")
        a = model(inputs_embeds=x).logits
        set_active(model, "scoring")
        s = model(inputs_embeds=x).logits
        set_active(model, ["asr", "scoring"])
        both = model(inputs_embeds=x).logits

    print(f"asr  vs scoring : {(a - s).abs().max().item():.4f}")
    print(f"both vs asr     : {(both - a).abs().max().item():.4f}")
    print(f"both vs scoring : {(both - s).abs().max().item():.4f}")


if __name__ == "__main__":
    main()