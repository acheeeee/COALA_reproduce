"""評估層共用工具：選 encoder 的精度、載入 Stage 1 / Stage 2 模型。"""

import torch

from model import COALAModel

AMP_CHOICES = ["auto", "bf16", "fp16", "fp32"]


def resolve_amp(name):
    """--amp 參數 → encoder 用的 autocast dtype（None = 不用 autocast，全 fp32）。

    auto：Ampere 以後（3090 / A5000，compute capability ≥ 8）用 bf16，
          更舊的卡（2080 Ti，7.5）沒有原生 bf16 → 用 fp16。
    """
    if name == "auto":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[name]


def _load_trainable(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not unexpected, f"{path} 有不認識的鍵: {unexpected[:5]}"
    return ck


def load_stage1_model(ckpt, stack, dev, amp="auto"):
    """Stage 1 模型（ASR，LoRA asr 啟用）。"""
    model = COALAModel(stack=stack)
    ck = _load_trainable(model, ckpt)
    model.amp_dtype = resolve_amp(amp)
    model.to(dev).eval()
    model.lm.set_adapter("asr")
    print(f"載入 Stage 1 {ckpt}（epoch {ck.get('epoch')}, step {ck.get('step')}）"
          f"，encoder 精度 {model.amp_dtype}")
    return model


def load_stage2_model(ckpt, dev, stage1_ckpt=None, amp="auto"):
    """Stage 2 模型（打分，LoRA scoring + projector）。

    Stage 2 的 checkpoint 只存了 projector + LoRA scoring，
    所以要先載入它當初用的 Stage 1 checkpoint（路徑記在 ck["args"] 裡）。
    """
    ck2 = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck2.get("args", {})
    stage1_ckpt = stage1_ckpt or a["stage1_ckpt"]
    model = COALAModel(stack=a.get("stack", 2))
    _load_trainable(model, stage1_ckpt)
    model.setup_stage2()
    _, unexpected = model.load_state_dict(ck2["model"], strict=False)
    assert not unexpected, f"{ckpt} 有不認識的鍵: {unexpected[:5]}"
    model.amp_dtype = resolve_amp(amp)
    model.to(dev).eval()
    print(f"載入 Stage 2 {ckpt}（loss={a.get('loss')}, epoch {ck2.get('epoch')}, step {ck2.get('step')}）"
          f"，Stage 1 = {stage1_ckpt}，encoder 精度 {model.amp_dtype}")
    return model
