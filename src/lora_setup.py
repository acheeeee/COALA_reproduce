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

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]


def make_lora_config():
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # TODO 1: 載入 lm = AutoModelForCausalLM.from_pretrained(LM_ID)
    #         印出總參數量（應該是 134.5M）

    # TODO 2: 用 get_peft_model(lm, make_lora_config(), adapter_name="asr")
    #         包起來，印出 model.print_trainable_parameters()

    # TODO 3: 用 model.add_adapter("scoring", make_lora_config()) 加第二組
    #         再印一次可訓練參數量

    # TODO 4: 用 model.set_adapter("asr") 切換，印出 count_trainable
    #         再 model.set_adapter("scoring")，印出 count_trainable
    #         兩個數字應該相同（兩組 config 一樣）

    # TODO 5: 跑一次 forward 確認能動
    #         x = torch.randn(1, 20, 576)
    #         out = model(inputs_embeds=x)
    #         印出 out.logits.shape（應該是 (1, 20, 49152)）
    ...


if __name__ == "__main__":
    main()